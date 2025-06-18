#!/usr/bin/env python3
"""
Debug script using your existing endo_datasplit_new.csv
This will help us find exactly where the memory allocation fails
"""

import torch
import pandas as pd
import os
import psutil
import zarr
import numpy as np

def print_memory_status(stage):
    """Print current memory usage"""
    memory = psutil.virtual_memory()
    print(f"\n--- {stage} ---")
    print(f"Available RAM: {memory.available/1024**3:.2f} GB")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_mem:.2f}/{gpu_total:.2f} GB")

def analyze_your_csv():
    """Analyze your existing CSV file"""
    csv_path = "endo_datasplit_new.csv"
    
    print("🔍 ANALYZING YOUR EXISTING CSV")
    print_memory_status("Initial")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Read and analyze CSV
    df = pd.read_csv(csv_path, header=None)
    print(f"\n📊 Dataset Statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Training entries: {len(df[df[0] == 'train'])}")
    print(f"Validation entries: {len(df[df[0] == 'validate'])}")
    
    # Check first few entries
    print(f"\n📁 First 3 entries:")
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"\nEntry {i+1}:")
        print(f"  Split: {row[0]}")
        print(f"  Input file: {row[1]}")
        print(f"  Input path: {row[2]}")
        print(f"  Target file: {row[3]}")
        print(f"  Target path: {row[4]}")
        
        # Check if files exist
        input_exists = os.path.exists(row[1])
        target_exists = os.path.exists(row[3])
        print(f"  Input exists: {input_exists}")
        print(f"  Target exists: {target_exists}")
        
        if input_exists:
            size_gb = os.path.getsize(row[1]) / 1024**3
            print(f"  Input file size: {size_gb:.2f} GB")
        
        if target_exists:
            size_gb = os.path.getsize(row[3]) / 1024**3
            print(f"  Target file size: {size_gb:.2f} GB")
    
    return True

def test_zarr_loading():
    """Test loading zarr files to see if they're the memory bottleneck"""
    csv_path = "endo_datasplit_new.csv"
    df = pd.read_csv(csv_path, header=None)
    
    print("\n🔄 TESTING ZARR FILE LOADING")
    
    # Test first training entry
    first_train = df[df[0] == 'train'].iloc[0]
    input_file = first_train[1]
    input_path = first_train[2]
    target_file = first_train[3] 
    target_path = first_train[4]
    
    print(f"Testing: {input_file}")
    print_memory_status("Before zarr loading")
    
    try:
        # Try to open zarr file
        print("Opening input zarr...")
        input_zarr = zarr.open(input_file, mode='r')
        print(f"✅ Input zarr opened successfully")
        print(f"Input zarr tree:\n{input_zarr.tree()}")
        
        print_memory_status("After opening input zarr")
        
        # Try to access the specific path
        print(f"Accessing path: {input_path}")
        input_array = input_zarr[input_path]
        print(f"Input array shape: {input_array.shape}")
        print(f"Input array dtype: {input_array.dtype}")
        print(f"Input array size: {np.prod(input_array.shape) * input_array.dtype.itemsize / 1024**3:.2f} GB")
        
        print_memory_status("After accessing input array")
        
        # Try loading a small chunk
        print("Loading small chunk (32x32x32)...")
        chunk = input_array[:32, :32, :32]
        print(f"Chunk loaded: {chunk.shape}")
        
        print_memory_status("After loading chunk")
        
        # Now test target
        print(f"\nTesting target: {target_file}")
        target_zarr = zarr.open(target_file, mode='r')
        print(f"✅ Target zarr opened successfully")
        
        target_array = target_zarr[target_path]
        print(f"Target array shape: {target_array.shape}")
        print(f"Target array dtype: {target_array.dtype}")
        print(f"Target array size: {np.prod(target_array.shape) * target_array.dtype.itemsize / 1024**3:.2f} GB")
        
        print_memory_status("After target loading")
        
        return True
        
    except Exception as e:
        print(f"❌ Zarr loading failed: {e}")
        print(f"Error type: {type(e)}")
        print_memory_status("After zarr error")
        return False

def test_dataloader_step_by_step():
    """Test dataloader creation step by step"""
    print("\n🔄 TESTING DATALOADER CREATION STEP BY STEP")
    
    print_memory_status("Before imports")
    
    # Import dataloader
    try:
        from cellmap_segmentation_challenge.utils import get_dataloader
        print("✅ Imports successful")
        print_memory_status("After imports")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test with minimal config
    try:
        print("Creating dataloader with minimal config...")
        train_loader, val_loader = get_dataloader(
            datasplit_path="endo_datasplit_new.csv",
            classes=["endo"],
            batch_size=1,
            input_array_info={"shape": (1, 32, 32), "scale": (8, 8, 8)},
            target_array_info={"shape": (1, 32, 32), "scale": (8, 8, 8)},
            spatial_transforms={},
            iterations_per_epoch=1,
            random_validation=False,
            device="cpu",  # Use CPU to isolate GPU memory issues
            use_mutual_exclusion=False,
            weighted_sampler=False,
        )
        
        print("✅ Dataloader created successfully!")
        print_memory_status("After dataloader creation")
        
        # Try to get one batch
        print("Getting first batch...")
        train_iter = iter(train_loader.loader)
        batch = next(train_iter)
        
        print("✅ First batch loaded successfully!")
        print(f"Batch keys: {list(batch.keys())}")
        for key, tensor in batch.items():
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        print_memory_status("After first batch")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataloader failed: {e}")
        print(f"Error type: {type(e)}")
        print_memory_status("After dataloader error")
        return False

def main():
    """Run complete debugging sequence"""
    print("🚨 DEBUGGING YOUR TRAINING MEMORY ISSUE")
    print("=" * 60)
    
    # Step 1: Analyze CSV
    if not analyze_your_csv():
        return
    
    # Step 2: Test zarr loading
    if not test_zarr_loading():
        print("\n❌ Problem identified: Zarr file loading")
        print("Your zarr files might be:")
        print("- Too large to fit in memory")
        print("- Corrupted")
        print("- Using too much memory per access")
        return
    
    # Step 3: Test dataloader
    if not test_dataloader_step_by_step():
        print("\n❌ Problem identified: Dataloader creation")
        print("The cellmap dataloader is causing the memory issue")
        return
    
    print("\n✅ ALL TESTS PASSED!")
    print("The issue might be with:")
    print("1. GPU memory (try device='cpu')")
    print("2. Model size (try smaller model)")
    print("3. Training loop accumulation")
    
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Try training with device='cpu'")
    print("2. Use the minimal ResNet model")
    print("3. Set iterations_per_epoch=1 initially")

if __name__ == "__main__":
    main()