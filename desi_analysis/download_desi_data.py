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
    
    'fastspecfit_bright_hp00': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp00.fits",
        'size_gb': 0.4,
        'description': "FastSpecFit main bright galaxies HP00 - SFR and stellar properties"
    },
    
    'fastspecfit_bright_hp01': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp01.fits",
        'size_gb': 4.2,
        'description': "FastSpecFit main bright galaxies HP01 - SFR and stellar properties"
    },
    
    'fastspecfit_bright_hp02': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp02.fits",
        'size_gb': 5.6,
        'description': "FastSpecFit main bright galaxies HP02 - SFR and stellar properties"
    },
    
    'fastspecfit_bright_hp03': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp03.fits",
        'size_gb': 1.7,
        'description': "FastSpecFit main bright galaxies HP03 - SFR and stellar properties"
    },
    
    'fastspecfit_bright_hp04': {
        'url': f"{BASE_URL}/fastspecfit/iron/v3.0/catalogs/fastspec-iron-main-bright-nside1-hp04.fits",
        'size_gb': 4.0,
        'description': "FastSpecFit main bright galaxies HP04 - SFR and stellar properties"
    },
    
    'lss_bgs_bright': {
        'url': f"{BASE_URL}/lss/iron/v1.1/LSScats/full/BGS_BRIGHT_full.dat.fits",
        'size_gb': 0.5,
        'description': "BGS bright galaxy LSS catalog"
    },
    
    'lss_elg': {
        'url': f"{BASE_URL}/lss/iron/v1.1/LSScats/full/ELG_LOPnotqso_full.dat.fits",
        'size_gb': 0.3,
        'description': "ELG LSS catalog"
    },
    
    'lss_lrg': {
        'url': f"{BASE_URL}/lss/iron/v1.1/LSScats/full/LRG_full.dat.fits",
        'size_gb': 0.2,
        'description': "LRG LSS catalog"
    },
    
    'lss_qso': {
        'url': f"{BASE_URL}/lss/iron/v1.1/LSScats/full/QSO_full.dat.fits",
        'size_gb': 0.2,
        'description': "QSO LSS catalog"
    }
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
    """Download a smaller sample of DESI DR1 VAC data for initial analysis"""
    print("=== Downloading Sample DESI DR1 VAC Data ===")
    print("Real observational data only - no mocks or simulations")
    
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    sample_datasets = ['fastspecfit_bright_hp01', 'lss_bgs_bright', 'lss_elg', 'lss_lrg', 'lss_qso']
    
    for dataset_key in sample_datasets:
        if dataset_key in DATASETS:
            filename = f"{dataset_key.replace('_', '_')}.fits"
            print(f"\nDownloading {DATASETS[dataset_key]['description']}")
            if download_file(DATASETS[dataset_key]['url'], filename, 
                            DATASETS[dataset_key]['size_gb']):
                verify_desi_data(filename)
    
    print("\n=== Sample download complete ===")
    print("Ready for LSS-AGN-SF feedback analysis")

def download_expanded_sample():
    """Download multiple HEALPix regions for increased sample size"""
    print("=== Downloading Expanded DESI DR1 FastSpecFit Sample ===")
    print("Real observational data only - no mocks or simulations")
    
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    hp_regions = ['hp00', 'hp01', 'hp02', 'hp03', 'hp04']
    
    for hp in hp_regions:
        dataset_key = f'fastspecfit_bright_{hp}'
        if dataset_key in DATASETS:
            filename = f"fastspecfit_bright_{hp}.fits"
            print(f"\nDownloading {DATASETS[dataset_key]['description']}")
            if download_file(DATASETS[dataset_key]['url'], filename, 
                            DATASETS[dataset_key]['size_gb']):
                verify_desi_data(filename)
    
    lss_datasets = ['lss_bgs_bright', 'lss_elg', 'lss_lrg', 'lss_qso']
    for dataset_key in lss_datasets:
        if dataset_key in DATASETS:
            filename = f"{dataset_key.replace('_', '_')}.fits"
            print(f"\nDownloading {DATASETS[dataset_key]['description']}")
            if download_file(DATASETS[dataset_key]['url'], filename, 
                            DATASETS[dataset_key]['size_gb']):
                verify_desi_data(filename)
    
    print("\n=== Expanded sample download complete ===")
    print("Expected sample increase: ~5-10x current size")

def download_full_data():
    """Download the complete DESI DR1 VAC dataset"""
    print("=== Downloading Full DESI DR1 VAC Data ===")
    print("Real observational data only - no mocks or simulations")
    print("Warning: This will download ~25 GB of data")
    
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    
    for dataset_key, dataset_info in DATASETS.items():
        filename = f"{dataset_key.replace('_', '_')}.fits"
        print(f"\nDownloading {dataset_info['description']}")
        if download_file(dataset_info['url'], filename, dataset_info['size_gb']):
            verify_desi_data(filename)
    
    print("\n=== Full download complete ===")
    print("Complete DESI DR1 VAC dataset ready for analysis")

if __name__ == "__main__":
    print("DESI DR1 VAC Data Downloader")
    print("Real observational data only - no mocks or simulations")
    
    download_expanded_sample()
