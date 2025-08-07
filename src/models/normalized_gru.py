# src/models/normalized_gru.py
# Ultra-simple GRU with proper normalization to prevent collapse
# RELEVANT FILES: hybrid.py, preprocessor.py, main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class NormalizedGRU(nn.Module):
    """Ultra-simple GRU with normalization"""
    def __init__(self, input_size=2, hidden_size=8, output_size=1, dropout=0.1):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Single small GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout if hidden_size > 1 else 0)
        
        # Light dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Normalized GRU initialized with {total_params} parameters")
        
        # Initialize weights carefully
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'fc.weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'fc.bias' in name:
                nn.init.constant_(param, 0.0)  # Start at 0 before activation
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Normalize input across sequence
        x_reshaped = x.reshape(-1, features)
        x_normed = self.input_norm(x_reshaped)
        x = x_normed.reshape(batch_size, seq_len, features)
        
        # GRU processing
        out, hidden = self.gru(x)
        
        # Use hidden state instead of last output
        out = hidden.squeeze(0)
        
        # Dropout
        out = self.dropout(out)
        
        # Linear output
        out = self.fc(out)
        
        # No activation - let the model learn the scale
        # We'll handle positivity constraint in the loss/postprocessing
        return out

class NormalizedGRUModel:
    """Wrapper for normalized GRU with stable training"""
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        self.model = NormalizedGRU(hidden_size=8, dropout=0.1)
        
        # AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001,  # Lower learning rate
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Huber loss - more robust than MSE
        self.criterion = nn.HuberLoss(delta=0.1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store normalization parameters
        self.target_mean = None
        self.target_std = None
        
    def prepare_sequences(self, returns, garch_vol, targets):
        """Prepare sequences for training"""
        sequences = []
        sequence_targets = []
        
        min_len = min(len(returns), len(garch_vol), len(targets))
        
        for i in range(self.sequence_length, min_len):
            # Normalize inputs
            ret_seq = returns[i-self.sequence_length:i]
            garch_seq = garch_vol[i-self.sequence_length:i]
            
            # Stack features
            seq = np.column_stack([ret_seq, garch_seq])
            sequences.append(seq)
            sequence_targets.append(targets.iloc[i] if hasattr(targets, 'iloc') else targets[i])
            
        return np.array(sequences), np.array(sequence_targets)
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets, 
              epochs=100, batch_size=32, max_train_size=None):
        
        # Limit training data if needed
        if max_train_size and len(train_sequences) > max_train_size:
            indices = np.random.choice(len(train_sequences), max_train_size, replace=False)
            indices.sort()  # Maintain temporal order
            train_sequences = train_sequences[indices]
            train_targets = train_targets[indices]
        
        # Normalize targets
        self.target_mean = train_targets.mean()
        self.target_std = train_targets.std() + 1e-8
        
        train_targets_norm = (train_targets - self.target_mean) / self.target_std
        val_targets_norm = (val_targets - self.target_mean) / self.target_std
        
        # Create data loaders
        train_data = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_targets_norm)
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        val_data = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_targets_norm)
        )
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        print(f"   Training with {len(train_sequences)} samples, {len(val_sequences)} validation samples")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        min_delta = 1e-6
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                
                # Handle dimension mismatch
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_y.dim() == 0:
                    batch_y = batch_y.unsqueeze(0)
                    
                loss = self.criterion(outputs, batch_y)
                
                # Add small regularization to prevent collapse
                output_std = outputs.std()
                if output_std < 0.1:  # Penalize if variance too low
                    variance_penalty = 0.01 * (0.1 - output_std) ** 2
                    loss = loss + variance_penalty
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_predictions = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x).squeeze()
                    
                    # Handle dimensions
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                        
                    val_loss += self.criterion(outputs, batch_y).item()
                    
                    # Denormalize for variance check
                    pred_denorm = outputs.cpu().numpy() * self.target_std + self.target_mean
                    val_predictions.extend(pred_denorm if pred_denorm.ndim > 0 else [pred_denorm])
            
            val_loss /= len(val_loader)
            val_pred_std = np.std(val_predictions) if len(val_predictions) > 1 else 0
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'target_mean': self.target_mean,
                    'target_std': self.target_std
                }, 'best_normalized_gru.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                      f"Std={val_pred_std:.6f}, LR={current_lr:.6f}")
            
            # Check for collapse
            if val_pred_std < 1e-6 and epoch > 5:
                print(f"   ⚠️ Warning: Model collapse detected at epoch {epoch}")
                # Reset from best checkpoint if available
                try:
                    checkpoint = torch.load('best_normalized_gru.pth', weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"   Restored from best checkpoint")
                    break
                except:
                    pass
            
            # Early stopping with minimum epochs
            if patience_counter >= patience and epoch >= 20:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        # Load best model
        try:
            checkpoint = torch.load('best_normalized_gru.pth', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_mean = checkpoint['target_mean']
            self.target_std = checkpoint['target_std']
            print(f"   Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.6f}")
        except:
            print("   Warning: Could not load best model")
        
        return {'train_loss': train_loss, 'val_loss': best_val_loss}
    
    def predict(self, sequences):
        """Make predictions with denormalization"""
        self.model.eval()
        with torch.no_grad():
            sequences = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(sequences).squeeze()
            
            # Denormalize
            if self.target_mean is not None and self.target_std is not None:
                predictions = predictions.cpu().numpy() * self.target_std + self.target_mean
            else:
                predictions = predictions.cpu().numpy()
            
            # Ensure positive (volatility constraint)
            predictions = np.maximum(predictions, 1e-8)
            
        return predictions
    
    def load_best_model(self):
        """Load the best saved model"""
        try:
            checkpoint = torch.load('best_normalized_gru.pth', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_mean = checkpoint['target_mean']
            self.target_std = checkpoint['target_std']
            print(f"   Loaded model with val loss {checkpoint['val_loss']:.6f}")
        except:
            print("   No saved model found")
        return self