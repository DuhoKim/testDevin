#!/usr/bin/env python3
"""
Examine the structure of downloaded DESI DR1 VAC data
to understand available columns for analysis
"""

import os
import pandas as pd
from astropy.io import fits
from astropy.table import Table

def examine_lss_data():
    """Examine LSS catalog structure"""
    print("=== Examining LSS Catalog Structure ===")
    
    lss_files = {
        'BGS_BRIGHT': 'lss_bgs_bright.fits',
        'ELG': 'lss_elg.fits', 
        'LRG': 'lss_lrg.fits',
        'QSO': 'lss_qso.fits'
    }
    
    for name, filename in lss_files.items():
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            print(f"\n{name} catalog ({filename}):")
            with fits.open(filepath) as hdul:
                print(f"  Number of HDUs: {len(hdul)}")
                for i, hdu in enumerate(hdul):
                    print(f"  HDU {i}: {hdu.header.get('EXTNAME', 'PRIMARY')}")
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        table = Table(hdu.data)
                        print(f"    Shape: {len(table)} rows, {len(table.colnames)} columns")
                        print(f"    Columns ({len(table.colnames)}): {table.colnames}")
                        
                        coord_cols = [col for col in table.colnames if any(x in col.upper() for x in ['RA', 'DEC', 'Z', 'REDSHIFT'])]
                        if coord_cols:
                            print(f"    Coordinate columns: {coord_cols}")
                        else:
                            print("    No obvious coordinate columns found")
                        
                        key_cols = table.colnames[:10]
                        sample_data = table[key_cols][:3]
                        print(f"    Sample data (first 3 rows, first 10 cols):")
                        for col in key_cols:
                            print(f"      {col}: {list(sample_data[col])}")

def examine_fastspec_data():
    """Examine FastSpecFit catalog structure"""
    print("\n=== Examining FastSpecFit Catalog Structure ===")
    
    filepath = os.path.join("data", "fastspecfit_bright_hp00.fits")
    if os.path.exists(filepath):
        print(f"FastSpecFit catalog (fastspecfit_bright_hp00.fits):")
        with fits.open(filepath) as hdul:
            print(f"  Number of HDUs: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                print(f"  HDU {i}: {hdu.header.get('EXTNAME', 'PRIMARY')}")
                if hasattr(hdu, 'data') and hdu.data is not None:
                    table = Table(hdu.data)
                    print(f"    Shape: {len(table)} rows, {len(table.colnames)} columns")
                    print(f"    Columns ({len(table.colnames)}): {table.colnames}")
                    
                    sfr_cols = [col for col in table.colnames if any(x in col.upper() for x in ['SFR', 'MSTAR', 'MASS'])]
                    if sfr_cols:
                        print(f"    SFR/Mass columns: {sfr_cols}")
                    
                    coord_cols = [col for col in table.colnames if any(x in col.upper() for x in ['RA', 'DEC', 'Z', 'REDSHIFT'])]
                    if coord_cols:
                        print(f"    Coordinate columns: {coord_cols}")

if __name__ == "__main__":
    examine_lss_data()
    examine_fastspec_data()
