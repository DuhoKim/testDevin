#!/usr/bin/env python3
"""
Download DESI DR1 VAC catalogs for LSS-AGN-SF feedback analysis
Only uses real observational data from DESI DR1
"""

import os
import requests
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pandas as pd

BASE_URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1"

DATASETS = {
    'agnqso': {
        'url': f"{BASE_URL}/agnqso/v1.0/agnqso_desi.fits",
        'size_gb': 8.7,
        'description': "DESI DR1 AGN/QSO catalog with AGN properties"
    },
    
    'fastspecfit_bright_hp01': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp01.fits",
        'size_gb': 4.2,
        'description': "FastSpecFit main bright galaxies HP01 - SFR and stellar properties"
    },
    
    'fastspecfit_dark_hp00': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-dark-nside1-hp00.fits", 
        'size_gb': 1.4,
        'description': "FastSpecFit main dark galaxies HP00 - SFR and stellar properties"
    },
    
    'stellar_mass_emline': {
        'url': f"{BASE_URL}/stellar-mass-emline/v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits",
        'size_gb': 52.3,
        'description': "Stellar masses and emission line measurements"
    }
}

LSS_CATALOGS = {
    'BGS_BRIGHT': f"{BASE_URL}/lss/guadalupe/v1.0/LSScats/full/BGS_BRIGHT_full.dat.fits",
    'ELG': f"{BASE_URL}/lss/guadalupe/v1.0/LSScats/full/ELG_LOPnotqso_full.dat.fits", 
    'LRG': f"{BASE_URL}/lss/guadalupe/v1.0/LSScats/full/LRG_full.dat.fits",
    'QSO': f"{BASE_URL}/lss/guadalupe/v1.0/LSScats/full/QSO_full.dat.fits"
}

def download_file(url, filename, expected_size_gb=None):
    """Download a file with progress tracking"""
    print(f"Downloading {filename} from {url}")
    
    if os.path.exists(filename):
        file_size_gb = os.path.getsize(filename) / (1024**3)
        print(f"File {filename} already exists ({file_size_gb:.1f} GB)")
        if expected_size_gb and abs(file_size_gb - expected_size_gb) < 0.1:
            print("File size matches expected, skipping download")
            return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded/(1024**3):.2f} GB)", end='')
        
        print(f"\nDownloaded {filename} successfully")
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def verify_desi_data(filename):
    """Verify this is real DESI DR1 observational data"""
    try:
        with fits.open(filename) as hdul:
            header = hdul[0].header
            print(f"\nVerifying {filename}:")
            print(f"  TELESCOP: {header.get('TELESCOP', 'N/A')}")
            print(f"  SURVEY: {header.get('SURVEY', 'N/A')}")
            print(f"  PROGRAM: {header.get('PROGRAM', 'N/A')}")
            print(f"  EXTNAME: {header.get('EXTNAME', 'N/A')}")
            
            if 'DESI' in str(header.get('TELESCOP', '')):
                print("  ✓ Confirmed DESI observational data")
                return True
            else:
                print("  ⚠ Could not confirm DESI origin")
                return False
                
    except Exception as e:
        print(f"Error verifying {filename}: {e}")
        return False

def download_sample_data():
    """Download smaller sample datasets for initial analysis"""
    print("=== Downloading DESI DR1 VAC Sample Data ===")
    
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    print("\n1. Downloading LSS catalogs...")
    for name, url in LSS_CATALOGS.items():
        filename = f"lss_{name.lower()}.fits"
        if download_file(url, filename):
            verify_desi_data(filename)
    
    print("\n2. Downloading FastSpecFit bright galaxies catalog (HP01)...")
    filename = "fastspecfit_bright_hp01.fits"
    if download_file(DATASETS['fastspecfit_bright_hp01']['url'], filename, 
                    DATASETS['fastspecfit_bright_hp01']['size_gb']):
        verify_desi_data(filename)
    
    print("\n=== Sample data download complete ===")
    print("Use download_full_data() to download complete datasets")

def download_full_data():
    """Download complete datasets (warning: very large files)"""
    print("=== Downloading Complete DESI DR1 VAC Data ===")
    print("WARNING: This will download ~70 GB of data")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return
    
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    for name, info in DATASETS.items():
        filename = f"{name}.fits"
        print(f"\nDownloading {info['description']}")
        if download_file(info['url'], filename, info['size_gb']):
            verify_desi_data(filename)

if __name__ == "__main__":
    print("DESI DR1 VAC Data Downloader")
    print("Real observational data only - no mocks or simulations")
    
    download_sample_data()
