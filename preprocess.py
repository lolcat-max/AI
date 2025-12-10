import pandas as pd
import numpy as np
import scipy.signal
import pickle
import os

# CONFIGURATION
INPUT_FILE = 'neural_dataset.csv'
OUTPUT_FILE = 'processed_neural_data.pkl'

def process_pipeline():
    print("==================================================")
    print("NEURAL DATASET POST-PROCESSING PIPELINE")
    print("==================================================")

    # 1. Load Raw Data
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: {INPUT_FILE} not found. Run the Main System in 'Live' mode first.")
        return

    print(f"[1] Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"    Loaded {len(df)} raw samples.")

    # 2. Signal Filtering (Savitzky-Golay)
    # Removes high-frequency noise while preserving signal peaks
    print("[2] Applying Savitzky-Golay Filter...")
    window_length = 51 # Must be odd
    poly_order = 3
    
    # Handle short datasets
    if len(df) < window_length:
        window_length = len(df) // 2 * 2 + 1
        
    df['clean_signal'] = scipy.signal.savgol_filter(
        df['synaptic_weight'].values, 
        window_length=window_length, 
        polyorder=poly_order
    )

    # 3. Normalization (Min-Max to 0.0 - 1.0)
    print("[3] Normalizing Signal...")
    min_val = df['clean_signal'].min()
    max_val = df['clean_signal'].max()
    df['norm_signal'] = (df['clean_signal'] - min_val) / (max_val - min_val + 1e-8)

    # 4. Feature Extraction: Derivatives (Delta)
    # Delta represents the 'Velocity' of the neural change
    print("[4] Computing Derivatives (Delta)...")
    df['delta'] = np.gradient(df['norm_signal'])

    # 5. Packaging
    print(f"[5] Encoding and Saving to {OUTPUT_FILE}...")
    
    # We save as a list of dicts for fast iteration in the generator
    export_data = df[['norm_signal', 'delta']].to_dict('records')
    
    stats = {
        'samples': len(df),
        'mean_signal': df['norm_signal'].mean(),
        'max_delta': df['delta'].max(),
        'min_delta': df['delta'].min()
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'data': export_data, 'stats': stats}, f)

    print("✓ DONE. Pipeline Complete.")
    print(f"  Stats: {stats}")

if __name__ == "__main__":
    process_pipeline()

