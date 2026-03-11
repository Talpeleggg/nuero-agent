import os
import tempfile
import mne
import pandas as pd

def process_neuro_data(uploaded_file, apply_denoise=False):
    """
    Ingests raw neuro-data, optionally denoises it, and converts it to optimized Parquet.
    Returns the Pandas DataFrame and the path to the Parquet file.
    """
    # 1. Create a secure temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save the uploaded file to disk (MNE requires a physical file path)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = uploaded_file.name.lower().split('.')[-1]
    
    # 2. Universal Parsing Logic
    if file_ext == 'edf':
        # Load the raw EDF file securely
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # REAL-TIME SCIENTIST USE CASE: Automated Denoising
        if apply_denoise:
            # Apply a 1-40Hz bandpass filter to remove line noise and low-frequency drift
            raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
            
        df = raw.to_data_frame()
        
    elif file_ext == 'csv':
        df = pd.read_csv(file_path)
        # Note: Advanced filtering on CSVs requires knowing the sampling rate.
        # We skip the MNE filter for plain CSVs in this iteration.
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please upload .edf or .csv")

    # 3. Compression Engine: Convert to Apache Parquet for memory efficiency
    parquet_path = os.path.join(temp_dir, "optimized_dataset.parquet")
    
    # Save as parquet (drastically reduces size and speeds up LLM processing)
    df.to_parquet(parquet_path, engine='fastparquet')
    
    return df, parquet_path