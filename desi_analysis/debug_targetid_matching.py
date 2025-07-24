#!/usr/bin/env python3
"""
Debug TARGETID matching between FastSpecFit and LSS catalogs
"""

import pandas as pd
from astropy.io import fits
from astropy.table import Table
import os

def debug_targetid_matching():
    """Debug why TARGETID matching is failing"""
    print("=== Debugging TARGETID Matching ===")
    
    bgs_file = os.path.join("data", "lss_bgs_bright.fits")
    if os.path.exists(bgs_file):
        print("Loading BGS LSS catalog...")
        with fits.open(bgs_file) as hdul:
            bgs_table = Table(hdul[1].data)
            bgs_names = [col for col in bgs_table.colnames if len(bgs_table[col].shape) <= 1]
            bgs_data = bgs_table[bgs_names].to_pandas()
            print(f"BGS data: {len(bgs_data)} objects")
            print(f"BGS TARGETID sample: {bgs_data['TARGETID'].head()}")
            print(f"BGS TARGETID type: {type(bgs_data['TARGETID'].iloc[0])}")
            print(f"BGS TARGETID range: {bgs_data['TARGETID'].min()} to {bgs_data['TARGETID'].max()}")
    
    fastspec_file = os.path.join("data", "fastspecfit_bright_hp01.fits")
    if os.path.exists(fastspec_file):
        print("\nLoading FastSpecFit catalog...")
        with fits.open(fastspec_file) as hdul:
            metadata_table = Table(hdul[1].data)
            metadata_names = [col for col in metadata_table.colnames if len(metadata_table[col].shape) <= 1]
            metadata_df = metadata_table[metadata_names].to_pandas()
            print(f"FastSpecFit metadata: {len(metadata_df)} objects")
            print(f"FastSpecFit TARGETID sample: {metadata_df['TARGETID'].head()}")
            print(f"FastSpecFit TARGETID type: {type(metadata_df['TARGETID'].iloc[0])}")
            print(f"FastSpecFit TARGETID range: {metadata_df['TARGETID'].min()} to {metadata_df['TARGETID'].max()}")
            
            bgs_targetids = set(bgs_data['TARGETID'])
            fastspec_targetids = set(metadata_df['TARGETID'])
            overlap = bgs_targetids.intersection(fastspec_targetids)
            print(f"\nOverlap analysis:")
            print(f"BGS unique TARGETIDs: {len(bgs_targetids)}")
            print(f"FastSpecFit unique TARGETIDs: {len(fastspec_targetids)}")
            print(f"Overlapping TARGETIDs: {len(overlap)}")
            
            if len(overlap) > 0:
                print(f"Sample overlapping TARGETIDs: {list(overlap)[:5]}")
            else:
                print("No overlapping TARGETIDs found!")
                
                print("\nChecking for data type conversion issues...")
                bgs_sample = bgs_data['TARGETID'].iloc[:1000]
                fastspec_sample = metadata_df['TARGETID'].iloc[:1000]
                
                bgs_int = bgs_sample.astype(int)
                fastspec_int = fastspec_sample.astype(int)
                overlap_int = set(bgs_int).intersection(set(fastspec_int))
                print(f"Overlap after int conversion: {len(overlap_int)}")

if __name__ == "__main__":
    debug_targetid_matching()
