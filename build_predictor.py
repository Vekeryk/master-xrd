"""
Build Standalone Predictor Executable
======================================
Creates predict.exe using PyInstaller.

Usage:
    python build_predictor.py

Output:
    dist/predict.exe - Standalone executable (no Python needed!)
    dist/predict/    - All dependencies bundled

The .exe can be copied anywhere and will work without Python installation.
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path


def build_executable():
    """Build standalone executable using PyInstaller."""

    print("=" * 70)
    print("BUILDING STANDALONE PREDICTOR")
    print("=" * 70)

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"✓ PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller not installed!")
        print("   Install with: pip install pyinstaller")
        sys.exit(1)

    # Check if model exists
    model_path = Path("checkpoints/dataset_1000_dl100_7d_curve_val_best_curve.pt")
    if not model_path.exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        print("   Train model first or specify different path")
        sys.exit(1)

    print(f"✓ Model checkpoint found: {model_path}")

    # Clean previous builds
    for path in ['build', 'dist', 'predict.spec']:
        if Path(path).exists():
            print(f"  Cleaning {path}...")
            if Path(path).is_file():
                Path(path).unlink()
            else:
                shutil.rmtree(path)

    # PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',                    # Single executable file
        '--name=predict',               # Output name: predict.exe
        '--console',                    # Show console output
        '--clean',                      # Clean cache
        '--add-data', f'{model_path}{os.pathsep}checkpoints',  # Bundle model
        'predict.py'
    ]

    print("\n🔨 Building executable...")
    print(f"   Command: {' '.join(cmd)}")
    print()

    # Run PyInstaller
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("\n❌ Build failed!")
        sys.exit(1)

    # Check output
    exe_name = 'predict.exe' if sys.platform == 'win32' else 'predict'
    exe_path = Path('dist') / exe_name

    if not exe_path.exists():
        print(f"\n❌ Executable not found: {exe_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print(f"\nExecutable: {exe_path}")
    print(f"Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("📦 Package contents:")
    print("   - PyTorch runtime")
    print("   - Model checkpoint")
    print("   - All dependencies")
    print()
    print("🚀 Usage:")
    print(f"   {exe_path} curve.txt params.txt")
    print()
    print("💡 You can copy predict.exe anywhere - no Python needed!")
    print("=" * 70)


def test_executable():
    """Test the built executable with sample data."""

    exe_name = 'predict.exe' if sys.platform == 'win32' else 'predict'
    exe_path = Path('dist') / exe_name

    if not exe_path.exists():
        print("❌ Executable not found. Build first!")
        return

    print("\n🧪 Testing executable...")

    # Create test curve
    import numpy as np
    test_curve = np.random.rand(661) * 1e-3 + 1e-5
    test_curve_path = Path('test_curve.txt')

    with open(test_curve_path, 'w') as f:
        for val in test_curve:
            f.write(f"{val:.6e}\n")

    print(f"   Created test curve: {test_curve_path}")

    # Run predictor
    test_output_path = Path('test_params.txt')
    result = subprocess.run(
        [str(exe_path), str(test_curve_path), str(test_output_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("   ✅ Prediction successful!")
        print(f"\n   Output:")
        with open(test_output_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    print(f"      {line.rstrip()}")

        # Cleanup
        test_curve_path.unlink()
        test_output_path.unlink()
    else:
        print(f"   ❌ Prediction failed!")
        print(f"   Error: {result.stderr}")


if __name__ == "__main__":
    build_executable()

    # Ask if user wants to test
    response = input("\n🧪 Test the executable? (y/n): ")
    if response.lower() == 'y':
        test_executable()
