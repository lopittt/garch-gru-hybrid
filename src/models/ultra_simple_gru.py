# src/models/ultra_simple_gru.py
# Ultra-simple GRU with <100 parameters to avoid overfitting
# RELEVANT FILES: hybrid.py, preprocessor.py, main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class UltraSimpleGRU(nn.Module):
    """Ultra-simple GRU with minimal parameters"""
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()
        
        # Single small GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Direct linear output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Ultra-simple GRU initialized with {total_params} parameters")
        
        # Better initialization for output layer
        # Initialize to output around 0.01 (typical volatility)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.fc.bias, 0.01)
        
    def forward(self, x):
        # GRU processing
        out, _ = self.gru(x)
        
        # Take last timestep
        out = out[:, -1, :]
        
        # Linear output
        out = self.fc(out)
        
        # Softplus activation - smoother than ReLU, always positive
        # log(1 + exp(x)) ensures positive output without hard cutoff
        out = F.softplus(out, beta=1.0) + 1e-8
        
        return out

class UltraSimpleGRUModel:
    """Wrapper for ultra-simple GRU"""
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        self.model = UltraSimpleGRU()
        
        # Simple optimizer with higher learning rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.01,
            weight_decay=1e-5  # Light regularization
        )
        
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_sequences(self, returns, garch_vol, targets):
        """Prepare sequences for training"""
        sequences = []
        sequence_targets = []
        
        min_len = min(len(returns), len(garch_vol), len(targets))
        
        for i in range(self.sequence_length, min_len):
            seq = np.column_stack([
                returns[i-self.sequence_length:i],
                garch_vol[i-self.sequence_length:i]
            ])
            sequences.append(seq)
            sequence_targets.append(targets.iloc[i] if hasattr(targets, 'iloc') else targets[i])
            
        return np.array(sequences), np.array(sequence_targets)
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets, 
              epochs=100, batch_size=32, max_train_size=None):
        
        # Don't limit training data for ultra-simple model
        if max_train_size and len(train_sequences) > max_train_size:
            indices = np.random.choice(len(train_sequences), max_train_size, replace=False)
            train_sequences = train_sequences[indices]
            train_targets = train_targets[indices]
        
        # Create data loaders
        train_data = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_targets)
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        val_data = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_targets)
        )
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        print(f"   Training with {len(train_sequences)} samples, {len(val_sequences)} validation samples")
        
        best_val_loss = float('inf')
        patience = 20
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
                    
                    if outputs.dim() == 0:
                        val_predictions.append(outputs.item())
                    else:
                        val_predictions.extend(outputs.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_pred_std = np.std(val_predictions) if len(val_predictions) > 1 else 0
            
            # Save best model
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, 'best_ultra_simple_gru.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                print(f"   Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Std: {val_pred_std:.6f}")
            
            # Early stopping (but not too early!)
            if patience_counter >= patience and epoch >= 10:  # Minimum 10 epochs
                print(f"   Early stopping at epoch {epoch}")
                break
        
        # Load best model
        checkpoint = torch.load('best_ultra_simple_gru.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.6f}")
        
        return {'train_loss': train_loss, 'val_loss': best_val_loss}
    
    def predict(self, sequences):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            sequences = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(sequences)
        return predictions.cpu().numpy()
    
    def load_best_model(self):
        """Load the best saved model"""
        try:
            checkpoint = torch.load('best_ultra_simple_gru.pth', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded model with val loss {checkpoint['val_loss']:.6f}")
        except:
            print("   No saved model found")
        return self