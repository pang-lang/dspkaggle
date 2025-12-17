#!/usr/bin/env python3
"""
Test script to verify the official train/test split implementation.
This will create new splits using the official VQA-RAD test set.
"""

import os
import json
import shutil
from pathlib import Path
from combined_preprocessing import create_combined_data_loaders

def backup_old_splits():
    """Backup existing split files."""
    split_path = Path('data_splits/vqa_rad_seed42.json')
    if split_path.exists():
        backup_path = Path('data_splits/vqa_rad_seed42_OLD_BACKUP.json')
        shutil.copy(split_path, backup_path)
        print(f"✓ Backed up old splits to: {backup_path}")
        split_path.unlink()
        print(f"✓ Deleted old split file to force regeneration")
    else:
        print("No existing split file found")

def test_official_splits():
    """Test the new official split implementation."""
    print("="*70)
    print("TESTING OFFICIAL VQA-RAD TRAIN/TEST SPLITS")
    print("="*70)
    print()
    
    # Backup and remove old splits
    backup_old_splits()
    print()
    
    # Create loaders with official splits
    print("Creating data loaders with official test split...")
    print()
    
    train_loader, val_loader, test_loader, answer_vocab, split_indices = create_combined_data_loaders(
        dataset_name="flaviagiammarino/vqa-rad",
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        use_official_test_split=True,  # Use official splits
        split_seed=42
    )
    
    print()
    print("="*70)
    print("DATASET SPLIT STATISTICS")
    print("="*70)
    print()
    print(f"Training samples:   {len(train_loader.dataset):>5}")
    print(f"Validation samples: {len(val_loader.dataset):>5}")
    print(f"Test samples:       {len(test_loader.dataset):>5}")
    print(f"{'─'*70}")
    print(f"Total samples:      {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):>5}")
    print()
    print(f"Answer vocabulary:  {len(answer_vocab):>5} unique answers")
    print()
    print(f"Training batches:   {len(train_loader):>5}")
    print(f"Validation batches: {len(val_loader):>5}")
    print(f"Test batches:       {len(test_loader):>5}")
    print()
    
    # Verify split info
    print("="*70)
    print("SPLIT CONFIGURATION")
    print("="*70)
    print()
    if 'use_official_test' in split_indices:
        print(f"Using official test split: {split_indices['use_official_test']}")
        if split_indices.get('train_size'):
            print(f"Original HF train size:    {split_indices['train_size']}")
        if split_indices.get('test_size'):
            print(f"Original HF test size:     {split_indices['test_size']}")
    print()
    
    # Test a batch from each loader
    print("="*70)
    print("TESTING DATA LOADING")
    print("="*70)
    print()
    
    train_batch = next(iter(train_loader))
    print(f"✓ Train batch loaded:")
    print(f"  - Image shape: {train_batch['image'].shape}")
    print(f"  - Question: '{train_batch['question']['text'][0][:60]}...'")
    print(f"  - Answer: '{train_batch['answer']['text'][0]}'")
    print()
    
    val_batch = next(iter(val_loader))
    print(f"✓ Validation batch loaded:")
    print(f"  - Image shape: {val_batch['image'].shape}")
    print(f"  - Question: '{val_batch['question']['text'][0][:60]}...'")
    print(f"  - Answer: '{val_batch['answer']['text'][0]}'")
    print()
    
    test_batch = next(iter(test_loader))
    print(f"✓ Test batch loaded:")
    print(f"  - Image shape: {test_batch['image'].shape}")
    print(f"  - Question: '{test_batch['question']['text'][0][:60]}...'")
    print(f"  - Answer: '{test_batch['answer']['text'][0]}'")
    print()
    
    # Verify expected totals
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    print()
    
    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    expected_total = 2244  # 1793 + 451
    
    if total == expected_total:
        print(f"✅ SUCCESS: Total samples = {total} (matches expected {expected_total})")
    else:
        print(f"❌ WARNING: Total samples = {total} (expected {expected_total})")
    
    # Check test set size
    if len(test_loader.dataset) == 451:
        print(f"✅ SUCCESS: Test set size = 451 (official test split)")
    else:
        print(f"❌ WARNING: Test set size = {len(test_loader.dataset)} (expected 451)")
    
    print()
    print("="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    
    return train_loader, val_loader, test_loader, answer_vocab, split_indices

if __name__ == "__main__":
    try:
        loaders = test_official_splits()
        print("\n🎉 Official splits are working correctly!")
        print("\nNext steps:")
        print("  1. Review the new split file: data_splits/vqa_rad_seed42.json")
        print("  2. Update your training scripts if needed")
        print("  3. The old splits are backed up in: data_splits/vqa_rad_seed42_OLD_BACKUP.json")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

