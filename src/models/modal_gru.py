# src/models/modal_gru.py
# Modal cloud training wrapper for GRU model
# Provides cloud-based training with automatic resource management
# RELEVANT FILES: gru.py, hybrid.py, main.py

import modal
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Modal app definition
app = modal.App("garch-gru-training")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "torch",
    "numpy",
    "scikit-learn"
])

# Mount the current directory to access our code
volume = modal.Volume.from_name("garch-gru-volume", create_if_missing=True)

class GRUNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[512, 256, 128], output_size=1, dropout=0.3):
        super().__init__()
        
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.gru3 = nn.GRU(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out, _ = self.gru2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out, _ = self.gru3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc(out[:, -1, :])
        return out

@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU for faster training
    volumes={"/cache": volume},
    timeout=1800  # 30 minutes timeout
)
def train_gru_on_modal(train_sequences, train_targets, val_sequences, val_targets, epochs=50, batch_size=32):
    """Train GRU model on Modal cloud with GPU acceleration"""
    
    start_time = time.time()
    print(f"Starting Modal training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training data shape: {train_sequences.shape}")
    print(f"Validation data shape: {val_sequences.shape}")
    
    # Initialize model
    model = GRUNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
    criterion = nn.MSELoss()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Prepare data loaders
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
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Model checkpointing and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model to volume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, '/cache/best_gru_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    checkpoint = torch.load('/cache/best_gru_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to CPU for return
    model.to('cpu')
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Modal training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    # Return model state dict and training results
    return {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': checkpoint['val_loss'],
        'training_time': training_time,
        'final_epoch': checkpoint['epoch']
    }

class ModalGRUModel:
    """Wrapper class that handles Modal cloud training"""
    
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        self.model = None
        self.training_results = None
        
    def prepare_sequences(self, returns, garch_forecasts, targets):
        """Prepare sequences using GARCH forecasts as input"""
        sequences = []
        sequence_targets = []
        
        min_len = min(len(returns), len(garch_forecasts), len(targets))
        
        for i in range(self.sequence_length, min_len):
            seq = np.column_stack([
                returns[i-self.sequence_length:i],
                garch_forecasts[i-self.sequence_length:i]
            ])
            sequences.append(seq)
            sequence_targets.append(targets[i])
            
        return np.array(sequences), np.array(sequence_targets)
    
    def train_on_modal(self, train_sequences, train_targets, val_sequences, val_targets, epochs=50, batch_size=32, max_train_size=500):
        """Train the model using Modal cloud compute"""
        
        # Limit training data as specified in paper
        if len(train_sequences) > max_train_size:
            indices = np.random.choice(len(train_sequences), max_train_size, replace=False)
            train_sequences = train_sequences[indices]
            train_targets = train_targets[indices]
        
        print(f"Starting Modal training with {len(train_sequences)} training samples")
        print("This will automatically shut down after training completes")
        
        # Call the Modal function
        try:
            with app.run():
                self.training_results = train_gru_on_modal.remote(
                    train_sequences, train_targets, val_sequences, val_targets, epochs, batch_size
                )
            
            # Create local model and load the trained weights
            self.model = GRUNetwork()
            self.model.load_state_dict(self.training_results['model_state_dict'])
            
            print(f"Modal training completed successfully!")
            print(f"Training time: {self.training_results['training_time']:.2f} seconds")
            print(f"Final validation loss: {self.training_results['best_val_loss']:.6f}")
            
            return self.training_results
            
        except Exception as e:
            print(f"Modal training failed: {e}")
            raise
    
    def predict(self, sequences):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        with torch.no_grad():
            sequences = torch.FloatTensor(sequences)
            predictions = self.model(sequences)
        return predictions.numpy()
    
    def get_training_time(self):
        """Get the training time from Modal execution"""
        if self.training_results:
            return self.training_results['training_time']
        return None