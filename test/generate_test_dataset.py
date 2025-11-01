#!/usr/bin/env python3
"""
Generate 1000-sample test dataset for fast approach comparison.
"""

import sys
sys.path.insert(0, '.')

from dataset_stratified_7d import generate_stratified_dataset

if __name__ == "__main__":
    print("Generating 1000-sample test dataset...")

    X, Y = generate_stratified_dataset(
        n_samples=1000,
        dl=100e-8,
        dataset_name="dataset_1000_dl100_7d"
    )

    print(f"\n✅ Generated dataset:")
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    print(f"   Saved to: datasets/dataset_1000_dl100_7d.pkl")
