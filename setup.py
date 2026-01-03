#!/usr/bin/env python3
"""
Setup script for Paraphraser & Humanizer
Installs all required dependencies
"""

import subprocess
import sys
import os


def install_requirements():
    """Install all required packages."""
    print("Installing required packages...")
    print("=" * 50)
    
    requirements = [
        "PyQt6==6.6.1",
        "nltk==3.8.1",
        "wordnet",
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("=" * 50)
    print("All packages installed successfully!")
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\nDownloading NLTK data files...")
    print("=" * 50)
    
    try:
        import nltk
        
        datasets = [
            'punkt',
            'wordnet',
            'averaged_perceptron_tagger',
            'stopwords',
            'omw-1.4'
        ]
        
        for dataset in datasets:
            print(f"Downloading {dataset}...")
            try:
                nltk.download(dataset, quiet=False)
                print(f"✓ {dataset} downloaded")
            except Exception as e:
                print(f"! {dataset} download: {e}")
        
        print("=" * 50)
        print("NLTK data downloaded!")
        return True
    
    except ImportError:
        print("NLTK not yet installed. Please install PyQt6 first.")
        return False


def create_shortcut():
    """Create a shortcut for easy launching (Windows only)."""
    if sys.platform == "win32":
        print("\nCreating desktop shortcut...")
        try:
            import win32com.client
            
            desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
            shortcut_path = os.path.join(desktop, 'Paraphraser.lnk')
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = os.path.join(os.path.dirname(__file__), 'main.py')
            shortcut.WorkingDirectory = os.path.dirname(__file__)
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("✓ Shortcut created on desktop!")
        except Exception as e:
            print(f"Could not create shortcut: {e}")


def main():
    """Main setup function."""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 48 + "║")
    print("║" + "  Paraphraser & Humanizer Setup  ".center(48) + "║")
    print("║" + " " * 48 + "║")
    print("╚" + "=" * 48 + "╝")
    print()
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please fix the errors above.")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("Warning: Could not download some NLTK data. Continuing anyway...")
    
    # Create shortcut (Windows only)
    if sys.platform == "win32":
        create_shortcut()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("=" * 50)
    print("\nTo run the application, execute:")
    print(f"  python main.py")
    print("\nOr on Windows, you can double-click the 'Paraphraser.lnk' shortcut.")
    print()


if __name__ == "__main__":
    main()
