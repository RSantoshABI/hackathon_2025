"""
Quick verification script to check if all app components are properly set up.
Run this before launching the app for the first time.
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if filepath.exists():
        print(f"✅ {description}: {filepath.name}")
        return True
    else:
        print(f"❌ {description}: {filepath.name} NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status"""
    if dirpath.exists() and dirpath.is_dir():
        print(f"✅ {description}: {dirpath.name}/")
        return True
    else:
        print(f"❌ {description}: {dirpath.name}/ NOT FOUND")
        return False

def check_imports():
    """Try to import required packages"""
    print("\n" + "="*60)
    print("CHECKING PYTHON PACKAGES")
    print("="*60)
    
    packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('scipy', 'SciPy')
    ]
    
    all_ok = True
    for package_name, display_name in packages:
        try:
            __import__(package_name)
            print(f"✅ {display_name} installed")
        except ImportError:
            print(f"❌ {display_name} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_optimization_modules():
    """Check if optimization modules are accessible"""
    print("\n" + "="*60)
    print("CHECKING OPTIMIZATION MODULES")
    print("="*60)
    
    app_dir = Path(__file__).parent
    opt_dir = app_dir.parent / 'optimization'
    
    if not opt_dir.exists():
        print(f"❌ Optimization directory not found at: {opt_dir}")
        return False
    
    modules = [
        'data_processor.py',
        'constraints.py',
        'optimization.py',
        '__init__.py'
    ]
    
    all_ok = True
    for module in modules:
        module_path = opt_dir / module
        if module_path.exists():
            print(f"✅ {module}")
        else:
            print(f"❌ {module} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_config():
    """Check if config.py is set up"""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    try:
        import config
        print(f"✅ config.py loaded")
        print(f"   DATA_DIR: {config.DATA_DIR}")
        
        # Check if data files exist
        data_files = [
            ('ELASTICITY_PATH', config.ELASTICITY_PATH),
            ('REFERENCE_PATH', config.REFERENCE_PATH),
            ('COMPETITOR_ELASTICITY_PATH', config.COMPETITOR_ELASTICITY_PATH),
            ('COMPETITOR_REFERENCE_PATH', config.COMPETITOR_REFERENCE_PATH),
            ('SEGMENT_MAPPING_PATH', config.SEGMENT_MAPPING_PATH)
        ]
        
        print("\n   Checking data files:")
        all_exist = True
        for name, path in data_files:
            if Path(path).exists():
                print(f"   ✅ {Path(path).name}")
            else:
                print(f"   ❌ {Path(path).name} NOT FOUND")
                all_exist = False
        
        if not all_exist:
            print("\n   ⚠️  WARNING: Some data files not found.")
            print("   Update DATA_DIR in config.py to point to your data location.")
        
        return True
    except Exception as e:
        print(f"❌ Error loading config.py: {e}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("PRICE OPTIMIZATION APP - SETUP VERIFICATION")
    print("="*60)
    
    app_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("CHECKING APP FILES")
    print("="*60)
    
    # Check app files
    app_files = [
        (app_dir / 'app.py', 'Main application'),
        (app_dir / 'config.py', 'Configuration file'),
        (app_dir / 'launch.py', 'Launcher script'),
        (app_dir / 'requirements.txt', 'Requirements file'),
        (app_dir / 'README.md', 'README'),
        (app_dir / 'USER_GUIDE.md', 'User guide')
    ]
    
    files_ok = all(check_file_exists(f, desc) for f, desc in app_files)
    
    # Check packages
    packages_ok = check_imports()
    
    # Check optimization modules
    modules_ok = check_optimization_modules()
    
    # Check config
    config_ok = check_config()
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if files_ok and packages_ok and modules_ok and config_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to launch the app:")
        print("  python launch.py")
        print("or")
        print("  streamlit run app.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before launching the app.")
        
        if not packages_ok:
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
        
        if not config_ok:
            print("\nTo configure data paths:")
            print("  Edit config.py and update DATA_DIR")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
