#!/usr/bin/env python3
"""
Setup verification script for YOLOv8 Object Detection App.
Run this to ensure all dependencies are properly installed.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor} detected")
        return True
    else:
        print(f"  ❌ Python 3.8+ required (found {version.major}.{version.minor})")
        return False

def check_imports():
    """Check if all required packages are installed"""
    print("\n✓ Checking required packages...")
    
    packages = {
        'streamlit': 'Streamlit',
        'ultralytics': 'Ultralytics',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'plotly': 'Plotly',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check if CUDA is available for GPU acceleration"""
    print("\n✓ Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("  ℹ️  No GPU detected (CPU will be used)")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not check GPU: {e}")
        return False

def check_models():
    """Check if YOLOv8 models can be loaded"""
    print("\n✓ Checking YOLOv8 models...")
    try:
        from ultralytics import YOLO
        print("  Attempting to load yolov8n model...")
        model = YOLO('yolov8n.pt')
        print("  ✅ YOLOv8n model loaded successfully")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not load model: {e}")
        print("     This is normal - model will download on first use")
        return False

def check_project_structure():
    """Check if all required files exist"""
    print("\n✓ Checking project structure...")
    
    required_files = [
        'app.py',
        'object_detection.py',
        'analytics.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("=" * 50)
    print("🎯 YOLOv8 Object Detection App - Setup Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("GPU Support", check_cuda),
        ("YOLOv8 Models", check_models),
        ("Project Structure", check_project_structure)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    
    for name, result in results.items():
        status = "✅" if result else "⚠️"
        print(f"{status} {name}")
    
    # Final verdict
    critical_ok = (results.get("Python Version", False) and 
                   results.get("Required Packages", False) and
                   results.get("Project Structure", False))
    
    print("\n" + "=" * 50)
    if critical_ok:
        print("✅ Setup looks good! You can run the app with:")
        print("   streamlit run app.py")
    else:
        print("❌ Please fix the issues above before running the app")
        print("   Run: pip install -r requirements.txt")
    
    print("=" * 50)
    
    return 0 if critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())
