# src/config/settings.py
# Configuration manager for GARCH-GRU simulation settings
# Centralizes all configuration including training location preferences
# RELEVANT FILES: config.json, hybrid.py, main.py

import json
import os
from pathlib import Path
from typing import Dict, Any, Union

class Settings:
    """Configuration manager for GARCH-GRU project settings"""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            # Look for config.json in project root
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / "config.json"
        
        self.config_file = Path(config_file)
        self._settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON configuration file"""
        if not self.config_file.exists():
            print(f"Warning: Configuration file {self.config_file} not found.")
            print("Using default settings.")
            return self._get_default_settings()
        
        try:
            with open(self.config_file, 'r') as f:
                settings = json.load(f)
            print(f"✅ Loaded configuration from {self.config_file}")
            return settings
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            print("Using default settings.")
            return self._get_default_settings()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default settings.")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Fallback default settings"""
        return {
            "training": {
                "location": "local",
                "modal_gpu": "T4",
                "epochs": 150,
                "batch_size": 500,
                "max_train_size": 500,
                "sequence_length": 6,
                "timeout_minutes": 30,
                "print_training_time": True,
                "verbose_training": True
            },
            "data": {
                "years_back": 10,
                "end_date": "2024-12-31",
                "gkyz_window": 10,
                "garch_p": 1,
                "garch_q": 1
            },
            "simulation": {
                "n_periods": 252,
                "n_paths": 100,
                "save_plots": True,
                "plot_dpi": 300
            },
            "modal": {
                "app_name": "garch-gru-training",
                "volume_name": "garch-gru-volume",
                "image_packages": ["torch", "numpy", "scikit-learn"]
            }
        }
    
    def get(self, key_path: str, default=None) -> Any:
        """Get a setting value using dot notation (e.g., 'training.location')"""
        keys = key_path.split('.')
        value = self._settings
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set a setting value using dot notation"""
        keys = key_path.split('.')
        current = self._settings
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save(self) -> None:
        """Save current settings to configuration file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
            print(f"✅ Settings saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def reload(self) -> None:
        """Reload settings from configuration file"""
        self._settings = self._load_settings()
    
    def print_current_config(self) -> None:
        """Print current configuration for debugging"""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        
        print(f"\n🎯 TRAINING SETTINGS:")
        print(f"   Location: {self.get('training.location')}")
        print(f"   Modal GPU: {self.get('training.modal_gpu')}")
        print(f"   Epochs: {self.get('training.epochs')}")
        print(f"   Batch size: {self.get('training.batch_size')}")
        print(f"   Max train size: {self.get('training.max_train_size')}")
        print(f"   Print training time: {self.get('training.print_training_time')}")
        
        print(f"\n📊 DATA SETTINGS:")
        print(f"   Years back: {self.get('data.years_back')}")
        print(f"   End date: {self.get('data.end_date')}")
        print(f"   GKYZ window: {self.get('data.gkyz_window')}")
        print(f"   GARCH(p,q): ({self.get('data.garch_p')},{self.get('data.garch_q')})")
        
        print(f"\n🎲 SIMULATION SETTINGS:")
        print(f"   Periods: {self.get('simulation.n_periods')}")
        print(f"   Paths: {self.get('simulation.n_paths')}")
        print(f"   Save plots: {self.get('simulation.save_plots')}")
        
        print("="*50)
        print("\n💡 TIP: See '_options' section in config.json for supported values")
    
    def print_options_help(self) -> None:
        """Print available options for all settings"""
        options = self._settings.get('_options', {})
        if not options:
            print("No options documentation found in config.json")
            return
            
        print("\n" + "="*60)
        print("CONFIGURATION OPTIONS & SUPPORTED VALUES")
        print("="*60)
        
        for section, params in options.items():
            section_name = section.upper().replace('_', ' ')
            print(f"\n📋 {section_name}:")
            
            for param, values in params.items():
                if isinstance(values, list):
                    if all(isinstance(v, bool) for v in values):
                        print(f"   {param}: {' | '.join(map(str, values))}")
                    else:
                        print(f"   {param}: {' | '.join(map(str, values))}")
                else:
                    print(f"   {param}: {values}")
        
        print("="*60)
    
    # Convenience properties for common settings
    @property
    def use_modal(self) -> bool:
        """Check if Modal training is enabled"""
        return self.get('training.location', 'local').lower() == 'modal'
    
    @property
    def training_location(self) -> str:
        """Get training location"""
        return self.get('training.location', 'local')
    
    @property
    def modal_gpu(self) -> str:
        """Get Modal GPU type"""
        return self.get('training.modal_gpu', 'T4')
    
    @property
    def print_training_time(self) -> bool:
        """Check if training time should be printed"""
        return self.get('training.print_training_time', True)

# Global settings instance
settings = Settings()

# Convenience functions
def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def reload_settings() -> None:
    """Reload settings from file"""
    global settings
    settings.reload()

def print_config() -> None:
    """Print current configuration"""
    settings.print_current_config()

# Example usage functions
def switch_to_modal() -> None:
    """Switch training to Modal cloud"""
    settings.set('training.location', 'modal')
    settings.save()
    print("✅ Switched to Modal cloud training")

def switch_to_local() -> None:
    """Switch training to local CPU"""
    settings.set('training.location', 'local') 
    settings.save()
    print("✅ Switched to local CPU training")

def set_modal_gpu(gpu_type: str) -> None:
    """Set Modal GPU type"""
    settings.set('training.modal_gpu', gpu_type)
    settings.save()
    print(f"✅ Set Modal GPU to {gpu_type}")

if __name__ == "__main__":
    # Test the settings system
    print("🧪 Testing Settings System")
    settings = Settings()
    settings.print_current_config()
    
    print(f"\nTest get: training.location = {settings.get('training.location')}")
    print(f"Test property: use_modal = {settings.use_modal}")
    print(f"Test property: modal_gpu = {settings.modal_gpu}")
    
    print("\n🔄 Testing configuration switches...")
    print(f"Current: {settings.training_location}")
    
    if settings.training_location == 'local':
        print("Switching to Modal...")
        switch_to_modal()
    else:
        print("Switching to Local...")
        switch_to_local()
    
    settings.reload()
    print(f"After switch: {settings.training_location}")