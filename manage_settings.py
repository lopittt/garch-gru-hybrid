#!/usr/bin/env python3
# manage_settings.py
# Convenient command-line interface for managing GARCH-GRU settings
# Allows easy switching between local and Modal training configurations

import sys
import os

# Add src to path
sys.path.append('src')
from config.settings import get_settings, switch_to_modal, switch_to_local, set_modal_gpu, print_config

def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    settings = get_settings()
    
    if command == 'show' or command == 'status':
        print_config()
        
    elif command == 'help' or command == 'options':
        settings.print_options_help()
        
    elif command == 'local':
        switch_to_local()
        print("✅ Training switched to local CPU")
        
    elif command == 'modal':
        gpu_type = sys.argv[2] if len(sys.argv) > 2 else 'T4'
        switch_to_modal()
        set_modal_gpu(gpu_type)
        print(f"✅ Training switched to Modal cloud with {gpu_type} GPU")
        
    elif command == 'gpu':
        if len(sys.argv) < 3:
            print("❌ Please specify GPU type (T4, A10G, A100, H100, L4)")
            return
        gpu_type = sys.argv[2]
        set_modal_gpu(gpu_type)
        print(f"✅ Modal GPU set to {gpu_type}")
        
    elif command == 'epochs':
        if len(sys.argv) < 3:
            print("❌ Please specify number of epochs")
            return
        try:
            epochs = int(sys.argv[2])
            settings.set('training.epochs', epochs)
            settings.save()
            print(f"✅ Training epochs set to {epochs}")
        except ValueError:
            print("❌ Invalid number of epochs")
            
    elif command == 'batch_size':
        if len(sys.argv) < 3:
            print("❌ Please specify batch size")
            return
        try:
            batch_size = int(sys.argv[2])
            settings.set('training.batch_size', batch_size)
            settings.save()
            print(f"✅ Batch size set to {batch_size}")
        except ValueError:
            print("❌ Invalid batch size")
            
    elif command == 'verbose':
        if len(sys.argv) < 3:
            current = settings.get('training.verbose_training', True)
            new_value = not current
        else:
            new_value = sys.argv[2].lower() in ['true', '1', 'yes', 'on']
        
        settings.set('training.verbose_training', new_value)
        settings.save()
        print(f"✅ Verbose training set to {new_value}")
        
    elif command == 'time':
        if len(sys.argv) < 3:
            current = settings.get('training.print_training_time', True)
            new_value = not current
        else:
            new_value = sys.argv[2].lower() in ['true', '1', 'yes', 'on']
        
        settings.set('training.print_training_time', new_value)
        settings.save()
        print(f"✅ Print training time set to {new_value}")
        
    elif command == 'paths':
        if len(sys.argv) < 3:
            print("❌ Please specify number of simulation paths")
            return
        try:
            n_paths = int(sys.argv[2])
            settings.set('simulation.n_paths', n_paths)
            settings.save()
            print(f"✅ Simulation paths set to {n_paths}")
        except ValueError:
            print("❌ Invalid number of paths")
            
    elif command == 'periods':
        if len(sys.argv) < 3:
            print("❌ Please specify number of simulation periods")
            return
        try:
            n_periods = int(sys.argv[2])
            settings.set('simulation.n_periods', n_periods)
            settings.save()
            print(f"✅ Simulation periods set to {n_periods}")
        except ValueError:
            print("❌ Invalid number of periods")
            
    else:
        print(f"❌ Unknown command: {command}")
        print_help()

def print_help():
    print("""
🔧 GARCH-GRU Settings Manager

USAGE:
    python manage_settings.py <command> [options]

COMMANDS:
    show              Show current configuration
    status            Same as show
    help              Show all supported configuration options
    options           Same as help
    
    local             Switch to local CPU training
    modal [gpu]       Switch to Modal cloud training (default: T4)
    gpu <type>        Set Modal GPU type (T4, A10G, A100, H100, L4)
    
    epochs <n>        Set training epochs
    batch_size <n>    Set batch size
    paths <n>         Set simulation paths
    periods <n>       Set simulation periods
    
    verbose [on/off]  Toggle verbose training output
    time [on/off]     Toggle training time printing

EXAMPLES:
    python manage_settings.py show
    python manage_settings.py modal A100
    python manage_settings.py local
    python manage_settings.py epochs 100
    python manage_settings.py gpu T4
    python manage_settings.py verbose off
    
For more help, see config.json or src/config/settings.py
""")

if __name__ == "__main__":
    main()