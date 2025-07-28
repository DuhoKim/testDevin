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
        self.gfinder_data = {}
    
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
    
    def load_gfinder_data(self):
        """Load Gfinder halo-based group catalog data"""
        print("Loading Gfinder halo-based group catalog...")
        
        gfinder_files = {
            'galaxy': 'gfinder_galaxy.fits',
            'group': 'gfinder_group.fits', 
            'gal2grp': 'gfinder_gal2grp.fits'
        }
        
        for catalog_name, filename in gfinder_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading Gfinder {catalog_name} catalog...")
                try:
                    with fits.open(filepath) as hdul:
                        if catalog_name == 'galaxy':
                            table = Table(hdul['GALAXY'].data)
                        elif catalog_name == 'group':
                            table = Table(hdul['GROUP'].data)
                        else:
                            table = Table(hdul['GAL2GRP'].data)
                        
                        column_names = [col for col in table.colnames if len(table[col].shape) <= 1]
                        self.gfinder_data[catalog_name] = table[column_names].to_pandas()
                        print(f"  {len(self.gfinder_data[catalog_name])} objects loaded")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                    continue
            else:
                print(f"  {filename} not found, skipping Gfinder {catalog_name} catalog")
        
        if all(key in self.gfinder_data for key in ['galaxy', 'group', 'gal2grp']):
            try:
                gal_grp = pd.merge(self.gfinder_data['gal2grp'], 
                                 self.gfinder_data['group'][['IGRP', 'RICH', 'GRP_LOGM']], 
                                 on='IGRP', how='left')
                
                self.gfinder_matched = pd.merge(self.gfinder_data['galaxy'], 
                                              gal_grp[['IGAL', 'RICH', 'GRP_LOGM', 'RANK']], 
                                              on='IGAL', how='left')
                
                self.gfinder_matched['RICH'] = self.gfinder_matched['RICH'].fillna(1)
                self.gfinder_matched['GRP_LOGM'] = self.gfinder_matched['GRP_LOGM'].fillna(11.0)
                
                print(f"  Gfinder matched catalog: {len(self.gfinder_matched)} galaxies")
                print(f"  Halo mass range: {self.gfinder_matched['GRP_LOGM'].min():.2f} to {self.gfinder_matched['GRP_LOGM'].max():.2f}")
                print(f"  Richness range: {self.gfinder_matched['RICH'].min()} to {self.gfinder_matched['RICH'].max()}")
                return True
            except Exception as e:
                print(f"  Error joining Gfinder catalogs: {e}")
                return False
        else:
            print("  Not all Gfinder catalogs loaded, skipping halo mass/richness analysis")
            return False

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
            
            if not hasattr(self, 'gfinder_matched'):
                self.load_gfinder_data()
            
            if hasattr(self, 'gfinder_matched'):
                print("Adding halo mass and richness indicators...")
                gfinder_coords = SkyCoord(ra=self.gfinder_matched['RA'].values*u.deg, 
                                        dec=self.gfinder_matched['DEC'].values*u.deg)
                lss_coords = SkyCoord(ra=data_clean['RA'].values*u.deg, 
                                    dec=data_clean['DEC'].values*u.deg)
                
                idx, d2d, d3d = lss_coords.match_to_catalog_sky(gfinder_coords)
                separation_cut = d2d < 1.0*u.arcsec
                
                data_clean['HALO_MASS'] = np.nan
                data_clean['RICHNESS'] = np.nan
                data_clean.loc[separation_cut, 'HALO_MASS'] = self.gfinder_matched.iloc[idx[separation_cut]]['GRP_LOGM'].values
                data_clean.loc[separation_cut, 'RICHNESS'] = self.gfinder_matched.iloc[idx[separation_cut]]['RICH'].values
                
                valid_halo = np.isfinite(data_clean['HALO_MASS'])
                valid_rich = np.isfinite(data_clean['RICHNESS'])
                
                if valid_halo.sum() > 0:
                    halo_percentiles = np.percentile(data_clean.loc[valid_halo, 'HALO_MASS'], [25, 75])
                    data_clean['HALO_ENVIRONMENT'] = 'INTERMEDIATE_HALO'
                    data_clean.loc[data_clean['HALO_MASS'] < halo_percentiles[0], 'HALO_ENVIRONMENT'] = 'LOW_HALO'
                    data_clean.loc[data_clean['HALO_MASS'] > halo_percentiles[1], 'HALO_ENVIRONMENT'] = 'HIGH_HALO'
                    
                if valid_rich.sum() > 0:
                    rich_percentiles = np.percentile(data_clean.loc[valid_rich, 'RICHNESS'], [25, 75])
                    data_clean['RICHNESS_ENVIRONMENT'] = 'INTERMEDIATE_RICH'
                    data_clean.loc[data_clean['RICHNESS'] < rich_percentiles[0], 'RICHNESS_ENVIRONMENT'] = 'LOW_RICH'
                    data_clean.loc[data_clean['RICHNESS'] > rich_percentiles[1], 'RICHNESS_ENVIRONMENT'] = 'HIGH_RICH'
                    
                print(f"Halo mass environment classification:")
                if 'HALO_ENVIRONMENT' in data_clean.columns:
                    print(data_clean['HALO_ENVIRONMENT'].value_counts())
                print(f"Richness environment classification:")
                if 'RICHNESS_ENVIRONMENT' in data_clean.columns:
                    print(data_clean['RICHNESS_ENVIRONMENT'].value_counts())
            
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
            env_cols = ['TARGETID', 'ENVIRONMENT', 'LOG_DENSITY']
            if 'HALO_ENVIRONMENT' in env_data.columns:
                env_cols.extend(['HALO_ENVIRONMENT', 'HALO_MASS'])
            if 'RICHNESS_ENVIRONMENT' in env_data.columns:
                env_cols.extend(['RICHNESS_ENVIRONMENT', 'RICHNESS'])
            if 'TIDAL_ENVIRONMENT' in env_data.columns:
                env_cols.extend(['TIDAL_ENVIRONMENT', 'LOG_TIDAL_FIELD'])
            
            merged = pd.merge(self.fastspec_data, 
                            env_data[env_cols], 
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
        
        if 'HALO_ENVIRONMENT' in merged_clean.columns:
            print("\nHalo mass environmental trends:")
            for env in ['LOW_HALO', 'INTERMEDIATE_HALO', 'HIGH_HALO']:
                env_data = merged_clean[merged_clean['HALO_ENVIRONMENT'] == env]
                if len(env_data) > 10:
                    median_sfr = np.median(env_data['LOG_SFR'])
                    median_ssfr = np.median(env_data['LOG_SSFR'])
                    print(f"  {env}: N={len(env_data)}, median log SFR={median_sfr:.2f}, median log sSFR={median_ssfr:.2f}")
        
        if 'RICHNESS_ENVIRONMENT' in merged_clean.columns:
            print("\nRichness environmental trends:")
            for env in ['LOW_RICH', 'INTERMEDIATE_RICH', 'HIGH_RICH']:
                env_data = merged_clean[merged_clean['RICHNESS_ENVIRONMENT'] == env]
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

    def create_combined_trend_plots(self, merged_clean):
        """Create combined plots that clearly show environmental trends across mass bins"""
        print("Creating combined trend visualization plots...")
        
        mass_bins = {
            'Low Mass\n(log M* < 10.0)': (0, 10.0),
            'Intermediate Mass\n(10.0 ≤ log M* < 11.0)': (10.0, 11.0),
            'High Mass\n(log M* ≥ 11.0)': (11.0, 15.0)
        }
        
        bin_data = {}
        for bin_name, (min_mass, max_mass) in mass_bins.items():
            if 'High Mass' in bin_name:
                mask = merged_clean['LOGMSTAR'] >= min_mass
            else:
                mask = ((merged_clean['LOGMSTAR'] >= min_mass) & 
                       (merged_clean['LOGMSTAR'] < max_mass))
            bin_data[bin_name] = merged_clean[mask].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        ax1 = axes[0, 0]
        environments = ['VOID', 'INTERMEDIATE', 'CLUSTER']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        x_positions = []
        box_data = []
        labels = []
        
        for i, env in enumerate(environments):
            for j, (bin_name, data) in enumerate(bin_data.items()):
                env_data = data[data['ENVIRONMENT'] == env]
                if len(env_data) > 10:
                    x_pos = i * 4 + j
                    x_positions.append(x_pos)
                    box_data.append(np.log10(env_data['SFR']))
                    labels.append(f"{env}\n{bin_name.split()[0]}")
        
        bp = ax1.boxplot(box_data, positions=x_positions, patch_artist=True, widths=0.6)
        
        mass_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(mass_colors[i % 3])
            patch.set_alpha(0.7)
        
        ax1.set_xticks([1, 5, 9])
        ax1.set_xticklabels(environments)
        ax1.set_ylabel('log SFR [M☉/yr]')
        ax1.set_title('Star Formation Rate vs Environment by Mass Bin')
        ax1.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=mass_colors[i], alpha=0.7, 
                                label=list(mass_bins.keys())[i].replace('\n', ' ')) 
                          for i in range(3)]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        ax2 = axes[0, 1]
        box_data_ssfr = []
        
        for i, env in enumerate(environments):
            for j, (bin_name, data) in enumerate(bin_data.items()):
                env_data = data[data['ENVIRONMENT'] == env]
                if len(env_data) > 10:
                    ssfr = env_data['SFR'] / (10**env_data['LOGMSTAR'])
                    log_ssfr = np.log10(ssfr + 1e-12)
                    box_data_ssfr.append(log_ssfr)
        
        bp2 = ax2.boxplot(box_data_ssfr, positions=x_positions, patch_artist=True, widths=0.6)
        
        for i, patch in enumerate(bp2['boxes']):
            patch.set_facecolor(mass_colors[i % 3])
            patch.set_alpha(0.7)
        
        ax2.set_xticks([1, 5, 9])
        ax2.set_xticklabels(environments)
        ax2.set_ylabel('log sSFR [yr⁻¹]')
        ax2.set_title('Specific SFR vs Environment by Mass Bin')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        mass_bin_names = ['Low', 'Intermediate', 'High']
        void_medians = []
        cluster_medians = []
        
        for bin_name, data in bin_data.items():
            void_data = data[data['ENVIRONMENT'] == 'VOID']
            cluster_data = data[data['ENVIRONMENT'] == 'CLUSTER']
            
            if len(void_data) > 10 and len(cluster_data) > 10:
                void_median = np.median(np.log10(void_data['SFR']))
                cluster_median = np.median(np.log10(cluster_data['SFR']))
                void_medians.append(void_median)
                cluster_medians.append(cluster_median)
            else:
                void_medians.append(np.nan)
                cluster_medians.append(np.nan)
        
        x_mass = np.arange(len(mass_bin_names))
        width = 0.35
        
        bars1 = ax3.bar(x_mass - width/2, void_medians, width, label='Void', 
                       color='lightblue', alpha=0.8)
        bars2 = ax3.bar(x_mass + width/2, cluster_medians, width, label='Cluster', 
                       color='lightcoral', alpha=0.8)
        
        ax3.set_xlabel('Stellar Mass Bin')
        ax3.set_ylabel('Median log SFR [M☉/yr]')
        ax3.set_title('Environmental Quenching by Mass Bin')
        ax3.set_xticks(x_mass)
        ax3.set_xticklabels(mass_bin_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        for i, (void_val, cluster_val) in enumerate(zip(void_medians, cluster_medians)):
            if not np.isnan(void_val) and not np.isnan(cluster_val):
                quench_strength = void_val - cluster_val
                ax3.text(i, max(void_val, cluster_val) + 0.1, 
                        f'Δ={quench_strength:.2f}', ha='center', fontweight='bold')
        
        ax4 = axes[1, 0]
        tidal_envs = ['LOW_TIDAL', 'INTERMEDIATE_TIDAL', 'HIGH_TIDAL']
        tidal_colors = ['purple', 'orange', 'red']
        
        for i, (bin_name, data) in enumerate(bin_data.items()):
            tidal_medians = []
            for tidal_env in tidal_envs:
                tidal_data = data[data['TIDAL_ENVIRONMENT'] == tidal_env]
                if len(tidal_data) > 10:
                    median_sfr = np.median(np.log10(tidal_data['SFR']))
                    tidal_medians.append(median_sfr)
                else:
                    tidal_medians.append(np.nan)
            
            ax4.plot(range(len(tidal_envs)), tidal_medians, 'o-', 
                    label=bin_name.split('\n')[0], linewidth=2, markersize=8)
        
        ax4.set_xlabel('Tidal Environment')
        ax4.set_ylabel('Median log SFR [M☉/yr]')
        ax4.set_title('SFR vs Tidal Field by Mass Bin')
        ax4.set_xticks(range(len(tidal_envs)))
        ax4.set_xticklabels(['Low Tidal', 'Intermediate', 'High Tidal'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = axes[1, 1]
        
        for i, (bin_name, data) in enumerate(bin_data.items()):
            void_prob_quartiles = np.percentile(data['VOID_PROB_R10'], [25, 50, 75])
            
            quartile_medians = []
            quartile_labels = ['Q1 (Dense)', 'Q2', 'Q3', 'Q4 (Void-like)']
            
            for j in range(4):
                if j == 0:
                    mask = data['VOID_PROB_R10'] <= void_prob_quartiles[0]
                elif j == 1:
                    mask = ((data['VOID_PROB_R10'] > void_prob_quartiles[0]) & 
                           (data['VOID_PROB_R10'] <= void_prob_quartiles[1]))
                elif j == 2:
                    mask = ((data['VOID_PROB_R10'] > void_prob_quartiles[1]) & 
                           (data['VOID_PROB_R10'] <= void_prob_quartiles[2]))
                else:
                    mask = data['VOID_PROB_R10'] > void_prob_quartiles[2]
                
                quartile_data = data[mask]
                if len(quartile_data) > 10:
                    median_sfr = np.median(np.log10(quartile_data['SFR']))
                    quartile_medians.append(median_sfr)
                else:
                    quartile_medians.append(np.nan)
            
            ax5.plot(range(4), quartile_medians, 'o-', 
                    label=bin_name.split('\n')[0], linewidth=2, markersize=8)
        
        ax5.set_xlabel('Void Probability Quartile')
        ax5.set_ylabel('Median log SFR [M☉/yr]')
        ax5.set_title('SFR vs Void Probability by Mass Bin')
        ax5.set_xticks(range(4))
        ax5.set_xticklabels(quartile_labels, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[1, 2]
        
        sample_sizes = []
        env_labels = []
        
        for env in environments:
            for bin_name in mass_bins.keys():
                data = bin_data[bin_name]
                env_data = data[data['ENVIRONMENT'] == env]
                sample_sizes.append(len(env_data))
                env_labels.append(f"{env}\n{bin_name.split()[0]}")
        
        bars = ax6.bar(range(len(sample_sizes)), sample_sizes, 
                      color=[colors[i//3] for i in range(len(sample_sizes))],
                      alpha=0.7)
        
        ax6.set_xlabel('Environment - Mass Bin')
        ax6.set_ylabel('Number of Galaxies')
        ax6.set_title('Sample Sizes by Environment and Mass')
        ax6.set_xticks(range(len(env_labels)))
        ax6.set_xticklabels([label.replace('\n', ' ') for label in env_labels], 
                           rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
            if size > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01,
                        f'{size:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('combined_environmental_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Generated: combined_environmental_trends.png")
        
        self.create_trend_summary_plot(bin_data, mass_bins)
    
    def create_trend_summary_plot(self, bin_data, mass_bins):
        """Create a focused summary plot of key environmental trends"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1 = axes[0]
        mass_centers = [9.5, 10.5, 11.5]  # Representative mass for each bin
        quenching_strengths = []
        quenching_errors = []
        
        for bin_name, data in bin_data.items():
            void_data = data[data['ENVIRONMENT'] == 'VOID']
            cluster_data = data[data['ENVIRONMENT'] == 'CLUSTER']
            
            if len(void_data) > 10 and len(cluster_data) > 10:
                void_sfr = np.log10(void_data['SFR'])
                cluster_sfr = np.log10(cluster_data['SFR'])
                
                quench_strength = np.median(void_sfr) - np.median(cluster_sfr)
                n_bootstrap = 100
                bootstrap_quench = []
                for _ in range(n_bootstrap):
                    void_boot = np.random.choice(void_sfr, size=len(void_sfr), replace=True)
                    cluster_boot = np.random.choice(cluster_sfr, size=len(cluster_sfr), replace=True)
                    bootstrap_quench.append(np.median(void_boot) - np.median(cluster_boot))
                
                quenching_strengths.append(quench_strength)
                quenching_errors.append(np.std(bootstrap_quench))
            else:
                quenching_strengths.append(np.nan)
                quenching_errors.append(np.nan)
        
        ax1.errorbar(mass_centers, quenching_strengths, yerr=quenching_errors,
                    fmt='o-', linewidth=3, markersize=10, capsize=5, capthick=2)
        ax1.set_xlabel('log Stellar Mass [M☉]')
        ax1.set_ylabel('Environmental Quenching Strength\n(log SFR_void - log SFR_cluster)')
        ax1.set_title('Quenching Strength vs Stellar Mass')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        for i, (mass, strength) in enumerate(zip(mass_centers, quenching_strengths)):
            if not np.isnan(strength):
                ax1.annotate(f'{strength:.2f}', (mass, strength), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontweight='bold', fontsize=12)
        
        ax2 = axes[1]
        environments = ['VOID', 'INTERMEDIATE', 'CLUSTER']
        x_env = np.arange(len(environments))
        width = 0.25
        
        mass_bin_names = ['Low Mass', 'Intermediate Mass', 'High Mass']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (bin_name, data) in enumerate(bin_data.items()):
            medians = []
            errors = []
            
            for env in environments:
                env_data = data[data['ENVIRONMENT'] == env]
                if len(env_data) > 10:
                    sfr_values = np.log10(env_data['SFR'])
                    medians.append(np.median(sfr_values))
                    errors.append(np.std(sfr_values) / np.sqrt(len(sfr_values)))
                else:
                    medians.append(np.nan)
                    errors.append(np.nan)
            
            ax2.errorbar(x_env + i*width, medians, yerr=errors,
                        fmt='o-', label=mass_bin_names[i], color=colors[i],
                        linewidth=2, markersize=8, capsize=3)
        
        ax2.set_xlabel('Environment')
        ax2.set_ylabel('Median log SFR [M☉/yr]')
        ax2.set_title('SFR vs Environment by Mass Bin')
        ax2.set_xticks(x_env + width)
        ax2.set_xticklabels(environments)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[2]
        
        for i, (bin_name, data) in enumerate(bin_data.items()):
            sf_fractions = []
            
            for env in environments:
                env_data = data[data['ENVIRONMENT'] == env]
                if len(env_data) > 10:
                    ssfr = env_data['SFR'] / (10**env_data['LOGMSTAR'])
                    log_ssfr = np.log10(ssfr + 1e-12)
                    sf_fraction = np.sum(log_ssfr > -11) / len(log_ssfr)
                    sf_fractions.append(sf_fraction)
                else:
                    sf_fractions.append(np.nan)
            
            ax3.plot(x_env, sf_fractions, 'o-', label=mass_bin_names[i], 
                    color=colors[i], linewidth=2, markersize=8)
        
        ax3.set_xlabel('Environment')
        ax3.set_ylabel('Star-Forming Fraction\n(log sSFR > -11)')
        ax3.set_title('Star-Forming Fraction vs Environment')
        ax3.set_xticks(x_env)
        ax3.set_xticklabels(environments)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('environmental_trends_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Generated: environmental_trends_summary.png")

def main():
    """Main analysis pipeline with stellar-mass binned analysis"""
    print("=== DESI DR1 LSS-AGN-SF Feedback Analysis ===")
    print("Using real observational data only")
    print("Expanded sample with multiple LSS indicators")
    
    analyzer = DESIAnalyzer()
    
    analyzer.load_data()
    analyzer.load_gfinder_data()
    
    sfr_results = analyzer.analyze_sfr_vs_environment()
    
    mass_binned_results = analyzer.analyze_stellar_mass_binned_environment()
    
    if mass_binned_results is not None:
        analyzer.create_combined_trend_plots(mass_binned_results)
    
    agn_results = analyzer.analyze_agn_vs_environment()
    
    print("\n=== Analysis Complete ===")
    print("Generated plots:")
    print("- sfr_vs_environment.png")
    print("- stellar_mass_binned_environment_*.png (for each mass bin)")
    print("- combined_environmental_trends.png (NEW: comprehensive overview)")
    print("- environmental_trends_summary.png (NEW: key trends summary)")
    print("- agn_vs_environment.png")

if __name__ == "__main__":
    main()
