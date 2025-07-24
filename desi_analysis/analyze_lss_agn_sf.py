#!/usr/bin/env python3
"""
Analyze feedback from LSS and AGN to star formation using DESI DR1 VAC data
Real observational data analysis only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import stats
from scipy.spatial import cKDTree
import os

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

class DESIAnalyzer:
    """Analyze DESI DR1 data for LSS-AGN-SF feedback"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.lss_data = {}
        self.fastspec_data = None
        self.agn_data = None
        self.stellar_data = None
        
    def load_data(self):
        """Load DESI DR1 VAC catalogs"""
        print("Loading DESI DR1 VAC data...")
        
        lss_files = {
            'BGS': 'lss_bgs_bright.fits',
            'ELG': 'lss_elg.fits', 
            'LRG': 'lss_lrg.fits',
            'QSO': 'lss_qso.fits'
        }
        
        for name, filename in lss_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {name} LSS catalog...")
                with fits.open(filepath) as hdul:
                    table = Table(hdul[1].data)
                    names = [col for col in table.colnames if len(table[col].shape) <= 1]
                    self.lss_data[name] = table[names].to_pandas()
                    print(f"  {len(self.lss_data[name])} objects loaded")
                    print(f"  Columns: {list(self.lss_data[name].columns)[:10]}...")  # Show first 10 columns
        
        fastspec_file = os.path.join(self.data_dir, "fastspecfit_bright_hp01.fits")
        if os.path.exists(fastspec_file):
            print("Loading FastSpecFit catalog...")
            with fits.open(fastspec_file) as hdul:
                metadata_table = Table(hdul[1].data)
                metadata_names = [col for col in metadata_table.colnames if len(metadata_table[col].shape) <= 1]
                metadata_df = metadata_table[metadata_names].to_pandas()
                
                specphot_table = Table(hdul[2].data)
                specphot_names = [col for col in specphot_table.colnames if len(specphot_table[col].shape) <= 1]
                specphot_df = specphot_table[specphot_names].to_pandas()
                
                self.fastspec_data = pd.merge(metadata_df, specphot_df, on='TARGETID', how='inner')
                print(f"  {len(self.fastspec_data)} spectra loaded")
                print(f"  Key columns available: RA, DEC, Z, LOGMSTAR, SFR")
                
                sfr_cols = [col for col in self.fastspec_data.columns if 'SFR' in col or 'LOGMSTAR' in col]
                print(f"  SFR/Mass columns: {sfr_cols}")
        
        agn_file = os.path.join(self.data_dir, "agnqso.fits")
        if os.path.exists(agn_file):
            print("Loading AGN/QSO catalog...")
            with fits.open(agn_file) as hdul:
                table = Table(hdul[1].data)
                names = [col for col in table.colnames if len(table[col].shape) <= 1]
                self.agn_data = table[names].to_pandas()
                print(f"  {len(self.agn_data)} AGN/QSO loaded")
                print(f"  Columns: {list(self.agn_data.columns)[:10]}...")  # Show first 10 columns
    
    def calculate_environment_metrics(self, catalog_name='BGS'):
        """Calculate large-scale structure environment metrics"""
        if catalog_name not in self.lss_data:
            print(f"LSS catalog {catalog_name} not available")
            return None
            
        data = self.lss_data[catalog_name]
        print(f"Calculating environment metrics for {catalog_name}...")
        
        if all(col in data.columns for col in ['RA', 'DEC', 'Z_not4clus']):
            valid_coords = np.isfinite(data['RA']) & np.isfinite(data['DEC']) & np.isfinite(data['Z_not4clus'])
            data_clean = data[valid_coords].copy()
            print(f"Using {len(data_clean)} objects with valid coordinates")
            
            coords = SkyCoord(ra=data_clean['RA'].values*u.deg, dec=data_clean['DEC'].values*u.deg)
            
            positions = np.column_stack([data_clean['RA'], data_clean['DEC'], data_clean['Z_not4clus']*3000])  # z*c/H0 approximation
            tree = cKDTree(positions)
            
            distances, indices = tree.query(positions, k=11)  # k=11 to exclude self
            density_metric = 1.0 / (distances[:, -1] + 1e-6)  # Inverse distance to 10th neighbor
            
            data_clean['DENSITY_METRIC'] = density_metric
            data_clean['LOG_DENSITY'] = np.log10(density_metric)
            
            density_percentiles = np.percentile(density_metric, [25, 75])
            data_clean['ENVIRONMENT'] = 'INTERMEDIATE'
            data_clean.loc[density_metric < density_percentiles[0], 'ENVIRONMENT'] = 'VOID'
            data_clean.loc[density_metric > density_percentiles[1], 'ENVIRONMENT'] = 'CLUSTER'
            
            print(f"Environment classification:")
            print(data_clean['ENVIRONMENT'].value_counts())
            
            return data_clean
        else:
            print("Required columns (RA, DEC, Z_not4clus) not found")
            return None
    
    def analyze_sfr_vs_environment(self):
        """Analyze star formation rate vs large-scale environment"""
        if self.fastspec_data is None:
            print("FastSpecFit data not loaded")
            return
            
        print("Analyzing SFR vs Environment...")
        
        env_data = self.calculate_environment_metrics('BGS')
        if env_data is None:
            return
            
        if 'TARGETID' in self.fastspec_data.columns and 'TARGETID' in env_data.columns:
            merged = pd.merge(self.fastspec_data, env_data[['TARGETID', 'ENVIRONMENT', 'LOG_DENSITY']], 
                            on='TARGETID', how='inner')
            print(f"Matched {len(merged)} objects between FastSpecFit and LSS")
        else:
            print("Cannot match catalogs - TARGETID not found")
            return
        
        if 'SFR' in merged.columns:
            sfr = merged['SFR']
        else:
            print("SFR column not found in FastSpecFit data")
            print(f"Available columns: {list(merged.columns)}")
            return
            
        if 'LOGMSTAR' in merged.columns:
            mstar = 10**merged['LOGMSTAR']  # Convert from log to linear
            ssfr = sfr / mstar
            merged['SSFR'] = ssfr
            merged['LOG_SSFR'] = np.log10(ssfr + 1e-12)
            merged['MSTAR'] = merged['LOGMSTAR']  # Keep log version for plotting
        else:
            print("LOGMSTAR column not found in FastSpecFit data")
            return
        
        valid = (sfr > 0) & (merged['LOGMSTAR'] > 0) & np.isfinite(merged['LOG_DENSITY'])
        merged_clean = merged[valid].copy()
        
        print(f"Valid objects for analysis: {len(merged_clean)}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1 = axes[0, 0]
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax1.hist(np.log10(env_data['SFR']), alpha=0.6, label=f'{env} (N={len(env_data)})', bins=30)
        ax1.set_xlabel('log SFR [M☉/yr]')
        ax1.set_ylabel('Number of Galaxies')
        ax1.set_title('Star Formation Rate vs Environment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax2.hist(env_data['LOG_SSFR'], alpha=0.6, label=f'{env} (N={len(env_data)})', bins=30)
        ax2.set_xlabel('log sSFR [yr⁻¹]')
        ax2.set_ylabel('Number of Galaxies')
        ax2.set_title('Specific Star Formation Rate vs Environment')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        scatter = ax3.scatter(merged_clean['LOG_DENSITY'], np.log10(merged_clean['SFR']), 
                            c=merged_clean['MSTAR'], alpha=0.5, s=1)
        ax3.set_xlabel('log Local Density')
        ax3.set_ylabel('log SFR [M☉/yr]')
        ax3.set_title('SFR vs Local Density (colored by stellar mass)')
        plt.colorbar(scatter, ax=ax3, label='log M* [M☉]')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        scatter = ax4.scatter(merged_clean['LOG_DENSITY'], merged_clean['LOG_SSFR'], 
                            c=merged_clean['MSTAR'], alpha=0.5, s=1)
        ax4.set_xlabel('log Local Density')
        ax4.set_ylabel('log sSFR [yr⁻¹]')
        ax4.set_title('sSFR vs Local Density (colored by stellar mass)')
        plt.colorbar(scatter, ax=ax4, label='log M* [M☉]')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sfr_vs_environment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== Statistical Analysis ===")
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 10:
                median_sfr = np.median(np.log10(env_data['SFR']))
                median_ssfr = np.median(env_data['LOG_SSFR'])
                print(f"{env}: median log SFR = {median_sfr:.2f}, median log sSFR = {median_ssfr:.2f}")
        
        return merged_clean
    
    def analyze_agn_vs_environment(self):
        """Analyze AGN properties vs large-scale environment"""
        if self.agn_data is None:
            print("AGN data not available")
            return
            
        print("Analyzing AGN vs Environment...")
        
        env_data = self.calculate_environment_metrics('QSO')
        if env_data is None:
            return
            
        if 'TARGETID' in self.agn_data.columns and 'TARGETID' in env_data.columns:
            merged = pd.merge(self.agn_data, env_data[['TARGETID', 'ENVIRONMENT', 'LOG_DENSITY']], 
                            on='TARGETID', how='inner')
            print(f"Matched {len(merged)} AGN with environment data")
        else:
            print("Cannot match AGN catalog with environment data")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1 = axes[0]
        env_counts = merged['ENVIRONMENT'].value_counts()
        ax1.bar(env_counts.index, env_counts.values)
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Number of AGN')
        ax1.set_title('AGN Count by Environment')
        ax1.grid(True, alpha=0.3)
        
        if 'L_BOL' in merged.columns:
            ax2 = axes[1]
            valid = np.isfinite(merged['L_BOL']) & (merged['L_BOL'] > 0)
            ax2.scatter(merged.loc[valid, 'LOG_DENSITY'], 
                       np.log10(merged.loc[valid, 'L_BOL']), alpha=0.6)
            ax2.set_xlabel('log Local Density')
            ax2.set_ylabel('log Bolometric Luminosity')
            ax2.set_title('AGN Luminosity vs Environment')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('agn_vs_environment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return merged

def main():
    """Main analysis pipeline"""
    print("=== DESI DR1 LSS-AGN-SF Feedback Analysis ===")
    print("Using real observational data only")
    
    analyzer = DESIAnalyzer()
    
    analyzer.load_data()
    
    sfr_results = analyzer.analyze_sfr_vs_environment()
    agn_results = analyzer.analyze_agn_vs_environment()
    
    print("\n=== Analysis Complete ===")
    print("Generated plots:")
    print("- sfr_vs_environment.png")
    print("- agn_vs_environment.png")

if __name__ == "__main__":
    main()
