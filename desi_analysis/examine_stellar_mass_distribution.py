#!/usr/bin/env python3
"""
Examine stellar mass distribution in FastSpecFit catalogs for optimal binning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os

def examine_stellar_mass_distribution():
    """Examine stellar mass distribution across multiple HEALPix regions"""
    print("=== Examining Stellar Mass Distribution ===")
    
    fastspec_files = [
        "fastspecfit_bright_hp00.fits",
        "fastspecfit_bright_hp01.fits", 
        "fastspecfit_bright_hp02.fits",
        "fastspecfit_bright_hp03.fits",
        "fastspecfit_bright_hp04.fits"
    ]
    
    all_masses = []
    all_sfr = []
    
    for fastspec_file in fastspec_files:
        filepath = os.path.join("data", fastspec_file)
        if os.path.exists(filepath):
            print(f"Loading {fastspec_file}...")
            with fits.open(filepath) as hdul:
                specphot_table = Table(hdul[2].data)
                specphot_names = [col for col in specphot_table.colnames if len(specphot_table[col].shape) <= 1]
                specphot_df = specphot_table[specphot_names].to_pandas()
                
                valid_mass = (specphot_df['LOGMSTAR'] > 0) & np.isfinite(specphot_df['LOGMSTAR'])
                valid_sfr = (specphot_df['SFR'] > 0) & np.isfinite(specphot_df['SFR'])
                valid = valid_mass & valid_sfr
                
                masses = specphot_df.loc[valid, 'LOGMSTAR']
                sfr = specphot_df.loc[valid, 'SFR']
                
                all_masses.extend(masses.values)
                all_sfr.extend(sfr.values)
                
                print(f"  Valid objects: {len(masses)}")
                print(f"  Mass range: {masses.min():.2f} to {masses.max():.2f} log M☉")
                print(f"  SFR range: {sfr.min():.3f} to {sfr.max():.3f} M☉/yr")
    
    if len(all_masses) == 0:
        print("No valid stellar mass data found")
        return
    
    all_masses = np.array(all_masses)
    all_sfr = np.array(all_sfr)
    
    print(f"\nCombined statistics:")
    print(f"Total valid objects: {len(all_masses)}")
    print(f"Mass range: {all_masses.min():.2f} to {all_masses.max():.2f} log M☉")
    print(f"Median mass: {np.median(all_masses):.2f} log M☉")
    print(f"Mean mass: {np.mean(all_masses):.2f} log M☉")
    
    mass_bins = {
        'low_mass': (0, 10.0),      
        'intermediate': (10.0, 11.0), 
        'high_mass': (11.0, 15.0)   
    }
    
    print(f"\nProposed stellar mass bins:")
    for bin_name, (min_mass, max_mass) in mass_bins.items():
        if bin_name == 'high_mass':
            count = np.sum(all_masses >= min_mass)
        else:
            count = np.sum((all_masses >= min_mass) & (all_masses < max_mass))
        print(f"  {bin_name}: {count} objects ({count/len(all_masses)*100:.1f}%)")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(all_masses, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(10.0, color='red', linestyle='--', label='Low/Intermediate boundary')
    ax1.axvline(11.0, color='red', linestyle='--', label='Intermediate/High boundary')
    ax1.set_xlabel('log Stellar Mass [M☉]')
    ax1.set_ylabel('Number of Galaxies')
    ax1.set_title('Stellar Mass Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(np.log10(all_sfr), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('log SFR [M☉/yr]')
    ax2.set_ylabel('Number of Galaxies')
    ax2.set_title('Star Formation Rate Distribution')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    scatter = ax3.scatter(all_masses, np.log10(all_sfr), alpha=0.1, s=1)
    ax3.set_xlabel('log Stellar Mass [M☉]')
    ax3.set_ylabel('log SFR [M☉/yr]')
    ax3.set_title('SFR vs Stellar Mass')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ssfr = all_sfr / (10**all_masses)
    ax4.hist(np.log10(ssfr), bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('log sSFR [yr⁻¹]')
    ax4.set_ylabel('Number of Galaxies')
    ax4.set_title('Specific Star Formation Rate Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stellar_mass_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGenerated: stellar_mass_distribution_analysis.png")

if __name__ == "__main__":
    examine_stellar_mass_distribution()
