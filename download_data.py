"""
Dataset Download Script
========================
Downloads the UCI Online Retail II dataset.
"""

import sys
import os
from pathlib import Path
import urllib.request
import ssl

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, RAW_DATA_FILE


def download_dataset():
    """Download the UCI Online Retail II dataset."""
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_FILE.exists():
        print(f"✅ Dataset already exists at: {RAW_DATA_FILE}")
        size_mb = RAW_DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        return
    
    # UCI ML Repository URL
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    zip_path = RAW_DATA_DIR / "online_retail_ii.zip"
    
    print("Downloading UCI Online Retail II dataset...")
    print(f"   URL: {url}")
    print(f"   Destination: {RAW_DATA_DIR}")
    
    # Create SSL context that doesn't verify (some corporate networks block UCI)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded / total_size * 100)
                bar_len = 40
                filled = int(bar_len * pct / 100)
                bar = '=' * filled + ' ' * (bar_len - filled)
                print(f"\r   [{bar}] {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end='')
            else:
                print(f"\r   Downloaded: {downloaded/1024/1024:.1f} MB", end='')
        
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print()  # New line after progress bar
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nAlternative: Please download manually:")
        print("   1. Go to: https://archive.ics.uci.edu/dataset/502/online+retail+ii")
        print("   2. Or Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        print(f"   3. Place the .xlsx file at: {RAW_DATA_FILE}")
        return
    
    # Extract zip
    print("Extracting...")
    import zipfile
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(RAW_DATA_DIR)
        
        # Find the xlsx file (might be in a subfolder)
        xlsx_files = list(RAW_DATA_DIR.rglob("*.xlsx"))
        
        if xlsx_files:
            # Move the first xlsx file to our expected location
            source = xlsx_files[0]
            if source != RAW_DATA_FILE:
                import shutil
                shutil.move(str(source), str(RAW_DATA_FILE))
                print(f"   Moved {source.name} -> {RAW_DATA_FILE.name}")
        
        # Clean up zip
        zip_path.unlink()
        
        # Clean up any empty extracted directories
        for d in RAW_DATA_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        
        size_mb = RAW_DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"Download complete! File size: {size_mb:.1f} MB")
        print(f"   Location: {RAW_DATA_FILE}")
        
    except zipfile.BadZipFile:
        # Maybe it was directly an xlsx file (some mirrors serve it differently)
        import shutil
        shutil.move(str(zip_path), str(RAW_DATA_FILE))
        
        if RAW_DATA_FILE.exists():
            size_mb = RAW_DATA_FILE.stat().st_size / (1024 * 1024)
            print(f"Download complete! File size: {size_mb:.1f} MB")
        else:
            print("Could not extract dataset. Please download manually.")


if __name__ == "__main__":
    download_dataset()
