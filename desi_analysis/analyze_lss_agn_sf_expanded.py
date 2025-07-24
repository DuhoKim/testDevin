#!/usr/bin/env python3
"""
Analyze feedback from large-scale structure and AGN to star formation
using DESI DR1 Value Added Catalogs with stellar-mass binned analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class DESIAnalyzer:
    """Analyze DESI DR1 data for LSS-AGN-SF feedback with stellar-mass binning"""
    
    def __init__(self, data_dir="data"):
        """Initialize analyzer with data directory"""
        self.data_dir = data_dir
        self.lss_data = {}
        self.fastspec_data = None
        self.agn_data = None
    
    def load_data(self):
        """Load DESI DR1 VAC data"""
        print("Loading DESI DR1 Value Added Catalogs...")
        
        lss_files = {
            'BGS': 'lss_bgs_bright.fits',
            'ELG': 'lss_elg.fits', 
            'LRG': 'lss_lrg.fits',
            'QSO': 'lss_qso.fits'
        }
        
        for catalog_name, filename in lss_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {catalog_name} LSS catalog...")
                with fits.open(filepath) as hdul:
                    table = Table(hdul[1].data)
                    column_names = [col for col in table.colnames if len(table[col].shape) <= 1]
                    self.lss_data[catalog_name] = table[column_names].to_pandas()
                    print(f"  {len(self.lss_data[catalog_name])} objects loaded")
                    print(f"  Columns: {list(self.lss_data[catalog_name].columns)[:10]}...")
        
        fastspec_files = [
            "fastspecfit_bright_hp00.fits",
            "fastspecfit_bright_hp01.fits", 
            "fastspecfit_bright_hp02.fits",
            "fastspecfit_bright_hp03.fits",
            "fastspecfit_bright_hp04.fits"
        ]
        
        all_fastspec_data = []
        
        for fastspec_file in fastspec_files:
            filepath = os.path.join(self.data_dir, fastspec_file)
            if os.path.exists(filepath):
                print(f"Loading {fastspec_file}...")
                with fits.open(filepath) as hdul:
                    metadata_table = Table(hdul[1].data)
                    metadata_names = [col for col in metadata_table.colnames if len(metadata_table[col].shape) <= 1]
                    metadata_df = metadata_table[metadata_names].to_pandas()
                    
                    specphot_table = Table(hdul[2].data)
                    specphot_names = [col for col in specphot_table.colnames if len(specphot_table[col].shape) <= 1]
                    specphot_df = specphot_table[specphot_names].to_pandas()
                    
                    merged_df = pd.merge(metadata_df, specphot_df, on='TARGETID', how='inner')
                    all_fastspec_data.append(merged_df)
                    print(f"  {len(merged_df)} spectra loaded from {fastspec_file}")
        
        if all_fastspec_data:
            self.fastspec_data = pd.concat(all_fastspec_data, ignore_index=True)
            print(f"Total FastSpecFit spectra loaded: {len(self.fastspec_data)}")
            print(f"  Key columns available: RA, DEC, Z, LOGMSTAR, SFR")
            
            sfr_cols = [col for col in self.fastspec_data.columns if 'SFR' in col or 'LOGMSTAR' in col]
            print(f"  SFR/Mass columns: {sfr_cols}")
        
        agn_file = os.path.join(self.data_dir, "agnqso.fits")
        if os.path.exists(agn_file):
            print("Loading AGN/QSO catalog...")
            with fits.open(agn_file) as hdul:
                agn_table = Table(hdul[1].data)
                agn_names = [col for col in agn_table.colnames if len(agn_table[col].shape) <= 1]
                self.agn_data = agn_table[agn_names].to_pandas()
                print(f"  {len(self.agn_data)} AGN loaded")

    def analyze_stellar_mass_distribution(self):
        """Analyze stellar mass distribution and create optimal bins"""
        if self.fastspec_data is None:
            print("FastSpecFit data not loaded")
            return None
            
        print("Analyzing stellar mass distribution...")
        
        valid_mass = (self.fastspec_data['LOGMSTAR'] > 0) & np.isfinite(self.fastspec_data['LOGMSTAR'])
        masses = self.fastspec_data.loc[valid_mass, 'LOGMSTAR']
        
        print(f"Valid stellar masses: {len(masses)} objects")
        print(f"Mass range: {masses.min():.2f} to {masses.max():.2f} log M☉")
        print(f"Median mass: {masses.median():.2f} log M☉")
        
        mass_bins = {
            'low_mass': (0, 10.0),      
            'intermediate': (10.0, 11.0), 
            'high_mass': (11.0, 15.0)   
        }
        
        for bin_name, (min_mass, max_mass) in mass_bins.items():
            if bin_name == 'high_mass':
                count = np.sum(masses >= min_mass)
            else:
                count = np.sum((masses >= min_mass) & (masses < max_mass))
            print(f"  {bin_name}: {count} objects ({count/len(masses)*100:.1f}%)")
        
        return mass_bins

    def calculate_void_probability(self, data_clean, sphere_radii=[5, 10, 15]):
        """Calculate void probability function for different sphere radii"""
        print("Calculating void probability function...")
        
        positions = np.column_stack([data_clean['RA'], data_clean['DEC'], data_clean['Z_not4clus']*3000])
        tree = cKDTree(positions)
        
        void_prob_metrics = {}
        
        for radius in sphere_radii:
            neighbor_counts = tree.query_ball_point(positions, radius, return_length=True)
            
            void_prob = 1.0 / (neighbor_counts + 1)  
            
            data_clean[f'VOID_PROB_R{radius}'] = void_prob
            void_prob_metrics[f'R{radius}'] = void_prob
            
            print(f"  Void probability R={radius}: mean={np.mean(void_prob):.3f}")
        
        return void_prob_metrics
    
    def calculate_tidal_field_proxy(self, data_clean):
        """Calculate tidal field proxy using local density gradient"""
        print("Calculating tidal field proxy...")
        
        positions = np.column_stack([data_clean['RA'], data_clean['DEC'], data_clean['Z_not4clus']*3000])
        tree = cKDTree(positions)
        
        distances_5, indices_5 = tree.query(positions, k=6)  
        distances_20, indices_20 = tree.query(positions, k=21)  
        
        density_5 = 1.0 / (distances_5[:, -1] + 1e-6)
        density_20 = 1.0 / (distances_20[:, -1] + 1e-6)
        
        tidal_proxy = np.log10(density_5 / density_20)
        
        data_clean['TIDAL_FIELD_PROXY'] = tidal_proxy
        data_clean['LOG_TIDAL_FIELD'] = tidal_proxy
        
        print(f"  Tidal field proxy: mean={np.mean(tidal_proxy):.3f}, std={np.std(tidal_proxy):.3f}")
        
        return tidal_proxy

    def calculate_environment_metrics(self, catalog_name='BGS'):
        """Calculate multiple large-scale structure environment metrics"""
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
            
            positions = np.column_stack([data_clean['RA'], data_clean['DEC'], data_clean['Z_not4clus']*3000])
            tree = cKDTree(positions)
            
            distances, indices = tree.query(positions, k=11)  
            density_metric = 1.0 / (distances[:, -1] + 1e-6)  
            
            data_clean['DENSITY_METRIC'] = density_metric
            data_clean['LOG_DENSITY'] = np.log10(density_metric)
            
            void_prob_metrics = self.calculate_void_probability(data_clean)
            tidal_field = self.calculate_tidal_field_proxy(data_clean)
            
            density_percentiles = np.percentile(density_metric, [25, 75])
            tidal_percentiles = np.percentile(tidal_field, [25, 75])
            
            data_clean['ENVIRONMENT'] = 'INTERMEDIATE'
            data_clean.loc[density_metric < density_percentiles[0], 'ENVIRONMENT'] = 'VOID'
            data_clean.loc[density_metric > density_percentiles[1], 'ENVIRONMENT'] = 'CLUSTER'
            
            data_clean['TIDAL_ENVIRONMENT'] = 'INTERMEDIATE_TIDAL'
            data_clean.loc[tidal_field < tidal_percentiles[0], 'TIDAL_ENVIRONMENT'] = 'LOW_TIDAL'
            data_clean.loc[tidal_field > tidal_percentiles[1], 'TIDAL_ENVIRONMENT'] = 'HIGH_TIDAL'
            
            print(f"Environment classification (density):")
            print(data_clean['ENVIRONMENT'].value_counts())
            print(f"Environment classification (tidal):")
            print(data_clean['TIDAL_ENVIRONMENT'].value_counts())
            
            return data_clean
        else:
            print("Required columns (RA, DEC, Z_not4clus) not found")
            return None

    def analyze_sfr_vs_environment(self):
        """Analyze star formation rates versus environment"""
        if self.fastspec_data is None:
            print("FastSpecFit data not loaded")
            return
            
        print("Analyzing SFR vs environment...")
        
        env_data = self.calculate_environment_metrics('BGS')
        if env_data is None:
            return
            
        if 'TARGETID' in self.fastspec_data.columns and 'TARGETID' in env_data.columns:
            merged = pd.merge(self.fastspec_data, 
                            env_data[['TARGETID', 'ENVIRONMENT', 'LOG_DENSITY']], 
                            on='TARGETID', how='inner')
            print(f"Matched {len(merged)} objects between FastSpecFit and LSS")
        else:
            print("Cannot match catalogs - TARGETID not found")
            return
        
        valid = (merged['SFR'] > 0) & (merged['LOGMSTAR'] > 0) & np.isfinite(merged['LOG_DENSITY'])
        merged_clean = merged[valid].copy()
        
        print(f"Valid objects for analysis: {len(merged_clean)}")
        
        merged_clean['SSFR'] = merged_clean['SFR'] / (10**merged_clean['LOGMSTAR'])
        merged_clean['LOG_SFR'] = np.log10(merged_clean['SFR'])
        ssfr_positive = merged_clean['SSFR'] > 0
        merged_clean = merged_clean[ssfr_positive].copy()
        merged_clean['LOG_SSFR'] = np.log10(merged_clean['SSFR'])
        
        print(f"After removing invalid sSFR: {len(merged_clean)} objects")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1 = axes[0, 0]
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax1.hist(env_data['LOG_SFR'], alpha=0.6, 
                        label=f'{env} (N={len(env_data)})', bins=30)
        ax1.set_xlabel('log SFR [M☉/yr]')
        ax1.set_ylabel('Number of Galaxies')
        ax1.set_title('Star Formation Rate Distribution by Environment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax2.hist(env_data['LOG_SSFR'], alpha=0.6, 
                        label=f'{env} (N={len(env_data)})', bins=30)
        ax2.set_xlabel('log sSFR [yr⁻¹]')
        ax2.set_ylabel('Number of Galaxies')
        ax2.set_title('Specific Star Formation Rate by Environment')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        scatter = ax3.scatter(merged_clean['LOG_DENSITY'], merged_clean['LOG_SFR'], 
                            c=merged_clean['LOGMSTAR'], alpha=0.5, s=1, cmap='viridis')
        ax3.set_xlabel('log Local Density')
        ax3.set_ylabel('log SFR [M☉/yr]')
        ax3.set_title('SFR vs Local Density (colored by stellar mass)')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('log M* [M☉]')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        scatter = ax4.scatter(merged_clean['LOG_DENSITY'], merged_clean['LOG_SSFR'], 
                            c=merged_clean['LOGMSTAR'], alpha=0.5, s=1, cmap='viridis')
        ax4.set_xlabel('log Local Density')
        ax4.set_ylabel('log sSFR [yr⁻¹]')
        ax4.set_title('sSFR vs Local Density (colored by stellar mass)')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('log M* [M☉]')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sfr_vs_environment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nEnvironmental trends in star formation:")
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = merged_clean[merged_clean['ENVIRONMENT'] == env]
            if len(env_data) > 10:
                median_sfr = np.median(env_data['LOG_SFR'])
                median_ssfr = np.median(env_data['LOG_SSFR'])
                print(f"  {env}: N={len(env_data)}, median log SFR={median_sfr:.2f}, median log sSFR={median_ssfr:.2f}")
        
        print("SFR vs environment analysis complete")
        print("Generated: sfr_vs_environment.png")
        
        return merged_clean

    def analyze_stellar_mass_binned_environment(self):
        """Analyze environmental effects in stellar mass bins"""
        if self.fastspec_data is None:
            print("FastSpecFit data not loaded")
            return
            
        print("Analyzing stellar-mass binned environmental effects...")
        
        mass_bins = self.analyze_stellar_mass_distribution()
        if mass_bins is None:
            return
        
        env_data = self.calculate_environment_metrics('BGS')
        if env_data is None:
            return
            
        if 'TARGETID' in self.fastspec_data.columns and 'TARGETID' in env_data.columns:
            merged = pd.merge(self.fastspec_data, 
                            env_data[['TARGETID', 'ENVIRONMENT', 'TIDAL_ENVIRONMENT', 
                                    'LOG_DENSITY', 'LOG_TIDAL_FIELD', 'VOID_PROB_R10']], 
                            on='TARGETID', how='inner')
            print(f"Matched {len(merged)} objects between FastSpecFit and LSS")
        else:
            print("Cannot match catalogs - TARGETID not found")
            return
        
        valid = ((merged['SFR'] > 0) & (merged['LOGMSTAR'] > 0) & 
                np.isfinite(merged['LOG_DENSITY']) & np.isfinite(merged['LOG_TIDAL_FIELD']))
        merged_clean = merged[valid].copy()
        
        # Calculate sSFR for mass-binned analysis
        merged_clean['SSFR'] = merged_clean['SFR'] / (10**merged_clean['LOGMSTAR'])
        merged_clean['LOG_SFR'] = np.log10(merged_clean['SFR'])
        ssfr_positive = merged_clean['SSFR'] > 0
        merged_clean = merged_clean[ssfr_positive].copy()
        merged_clean['LOG_SSFR'] = np.log10(merged_clean['SSFR'])
        
        print(f"Valid objects for analysis: {len(merged_clean)}")
        
        for bin_name, (min_mass, max_mass) in mass_bins.items():
            if bin_name == 'high_mass':
                mask = merged_clean['LOGMSTAR'] >= min_mass
            else:
                mask = ((merged_clean['LOGMSTAR'] >= min_mass) & 
                       (merged_clean['LOGMSTAR'] < max_mass))
            
            bin_data = merged_clean[mask]
            print(f"\n=== {bin_name.upper()} ANALYSIS ({len(bin_data)} objects) ===")
            
            if len(bin_data) > 50:  
                self.plot_mass_binned_environment(bin_data, bin_name)
                self.analyze_mass_bin_statistics(bin_data, bin_name)
        
        return merged_clean

    def plot_mass_binned_environment(self, data, bin_name):
        """Create plots for stellar-mass binned environmental analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax1 = axes[0, 0]
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = data[data['ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax1.hist(np.log10(env_data['SFR']), alpha=0.6, 
                        label=f'{env} (N={len(env_data)})', bins=20)
        ax1.set_xlabel('log SFR [M☉/yr]')
        ax1.set_ylabel('Number of Galaxies')
        ax1.set_title(f'SFR vs Density Environment ({bin_name})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for env in ['LOW_TIDAL', 'INTERMEDIATE_TIDAL', 'HIGH_TIDAL']:
            env_data = data[data['TIDAL_ENVIRONMENT'] == env]
            if len(env_data) > 0:
                ax2.hist(np.log10(env_data['SFR']), alpha=0.6, 
                        label=f'{env} (N={len(env_data)})', bins=20)
        ax2.set_xlabel('log SFR [M☉/yr]')
        ax2.set_ylabel('Number of Galaxies')
        ax2.set_title(f'SFR vs Tidal Environment ({bin_name})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        scatter = ax3.scatter(data['VOID_PROB_R10'], np.log10(data['SFR']), 
                            alpha=0.5, s=1)
        ax3.set_xlabel('Void Probability (R=10)')
        ax3.set_ylabel('log SFR [M☉/yr]')
        ax3.set_title(f'SFR vs Void Probability ({bin_name})')
        ax3.grid(True, alpha=0.3)
        
        data['SSFR'] = data['SFR'] / (10**data['LOGMSTAR'])
        valid_ssfr = data['SSFR'] > 0
        data = data[valid_ssfr].copy()
        data['LOG_SSFR'] = np.log10(data['SSFR'])
        
        ax4 = axes[1, 0]
        scatter = ax4.scatter(data['LOG_DENSITY'], data['LOG_SSFR'], 
                            alpha=0.5, s=1)
        ax4.set_xlabel('log Local Density')
        ax4.set_ylabel('log sSFR [yr⁻¹]')
        ax4.set_title(f'sSFR vs Local Density ({bin_name})')
        ax4.grid(True, alpha=0.3)
        
        ax5 = axes[1, 1]
        scatter = ax5.scatter(data['LOG_TIDAL_FIELD'], data['LOG_SSFR'], 
                            alpha=0.5, s=1)
        ax5.set_xlabel('log Tidal Field Proxy')
        ax5.set_ylabel('log sSFR [yr⁻¹]')
        ax5.set_title(f'sSFR vs Tidal Field ({bin_name})')
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[1, 2]
        scatter = ax6.scatter(data['VOID_PROB_R10'], data['LOG_SSFR'], 
                            alpha=0.5, s=1)
        ax6.set_xlabel('Void Probability (R=10)')
        ax6.set_ylabel('log sSFR [yr⁻¹]')
        ax6.set_title(f'sSFR vs Void Probability ({bin_name})')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'stellar_mass_binned_environment_{bin_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_mass_bin_statistics(self, data, bin_name):
        """Analyze statistics for stellar mass bin"""
        print(f"Statistical analysis for {bin_name}:")
        
        for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
            env_data = data[data['ENVIRONMENT'] == env]
            if len(env_data) > 10:
                median_sfr = np.median(np.log10(env_data['SFR']))
                median_ssfr = np.median(env_data['LOG_SSFR'])
                print(f"  {env}: N={len(env_data)}, median log SFR={median_sfr:.2f}, median log sSFR={median_ssfr:.2f}")
        
        for env in ['LOW_TIDAL', 'INTERMEDIATE_TIDAL', 'HIGH_TIDAL']:
            env_data = data[data['TIDAL_ENVIRONMENT'] == env]
            if len(env_data) > 10:
                median_sfr = np.median(np.log10(env_data['SFR']))
                median_ssfr = np.median(env_data['LOG_SSFR'])
                print(f"  {env}: N={len(env_data)}, median log SFR={median_sfr:.2f}, median log sSFR={median_ssfr:.2f}")

    def analyze_agn_vs_environment(self):
        """Analyze AGN properties versus environment"""
        if not hasattr(self, 'agn_data') or self.agn_data is None:
            print("AGN data not loaded - skipping AGN analysis")
            return
            
        print("Analyzing AGN properties vs environment...")
        
        env_data = self.calculate_environment_metrics('BGS')
        if env_data is None:
            return
            
        if 'TARGETID' in self.agn_data.columns and 'TARGETID' in env_data.columns:
            merged = pd.merge(self.agn_data, 
                            env_data[['TARGETID', 'ENVIRONMENT', 'LOG_DENSITY']], 
                            on='TARGETID', how='inner')
            print(f"Matched {len(merged)} AGN with environment data")
        else:
            print("Cannot match AGN and environment catalogs - TARGETID not found")
            return
        
        if len(merged) < 10:
            print("Insufficient AGN matches for analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1 = axes[0]
        env_counts = merged['ENVIRONMENT'].value_counts()
        ax1.bar(env_counts.index, env_counts.values)
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Number of AGN')
        ax1.set_title('AGN Distribution by Environment')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        if 'Z' in merged.columns:
            for env in ['VOID', 'INTERMEDIATE', 'CLUSTER']:
                env_data = merged[merged['ENVIRONMENT'] == env]
                if len(env_data) > 0:
                    ax2.hist(env_data['Z'], alpha=0.6, label=f'{env} (N={len(env_data)})', bins=20)
            ax2.set_xlabel('Redshift')
            ax2.set_ylabel('Number of AGN')
            ax2.set_title('AGN Redshift Distribution by Environment')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('agn_vs_environment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("AGN environment analysis complete")
        print("Generated: agn_vs_environment.png")

def main():
    """Main analysis pipeline with stellar-mass binned analysis"""
    print("=== DESI DR1 LSS-AGN-SF Feedback Analysis ===")
    print("Using real observational data only")
    print("Expanded sample with multiple LSS indicators")
    
    analyzer = DESIAnalyzer()
    
    analyzer.load_data()
    
    sfr_results = analyzer.analyze_sfr_vs_environment()
    
    mass_binned_results = analyzer.analyze_stellar_mass_binned_environment()
    
    agn_results = analyzer.analyze_agn_vs_environment()
    
    print("\n=== Analysis Complete ===")
    print("Generated plots:")
    print("- sfr_vs_environment.png")
    print("- stellar_mass_binned_environment_*.png (for each mass bin)")
    print("- agn_vs_environment.png")

if __name__ == "__main__":
    main()
