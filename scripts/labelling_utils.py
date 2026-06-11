#!/usr/bin/env python3
import pandas as pd
import glob
import os

def get_method_and_model(filename):
    """Extract method and model from filename pattern.
    Adapted from create_second_labelling_sample.py for robustness.
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    
    method = parts[0]
    
    model_name_str = "unknown_model" # Default
    
    # Try to find timestamp to delimit model name accurately
    timestamp_idx = -1
    # Common timestamp patterns: YYYYMMDD, or parts like 'all_articles_YYYYMMDD'
    # Example: method_modelname_all_articles_20250411_165119.csv -> modelname is parts[1:-4]
    # Example: method_modelname_20250411_165119.csv -> modelname is parts[1:-3]

    # Heuristic to find where model name ends and date/timestamp begins
    # Iterate backwards from typical timestamp location
    possible_ts_starts = []
    for i, part in enumerate(parts):
        if part.startswith("202") and len(part) >= 4 and part[:3].isdigit(): # e.g. 2023...
            if i > 0 and parts[i-1] == "articles" and i > 1 and parts[i-2] == "all": # ..._all_articles_YYYY...
                 possible_ts_starts.append(i-2)
            else: # ..._YYYY...
                 possible_ts_starts.append(i)
            # break # Take the first one found (usually the main date)

    if possible_ts_starts:
        timestamp_idx = min(possible_ts_starts) # Use the earliest potential start of a non-model part

    if timestamp_idx != -1 and timestamp_idx > 1: # parts[0] is method, parts[1:timestamp_idx] is model
        model_name_str = '_'.join(parts[1:timestamp_idx])
    elif len(parts) > 3 and parts[-1].endswith(".csv") and parts[-2].isdigit() and parts[-3].isdigit():
        # Fallback for structure like: method_model_name_YYYYMMDD_HHMMSS.csv
        model_name_str = '_'.join(parts[1:-3])
    elif len(parts) > 2 and parts[-1].endswith(".csv"): # Fallback if only .csv at end
         model_name_str = '_'.join(parts[1:-1])
    elif len(parts) > 1 : # Fallback if just method_model.csv
        model_name_str = parts[1].replace(".csv", "")
    else: # True fallback
        model_name_str = "unknown_model"


    # Map to short model names for categorization
    if 'Llama_3_1_8B' in model_name_str or 'Llama-3.1-8B' in model_name_str or 'llama8b' in model_name_str:
        model = 'llama8b'
    elif 'Llama_3_1_70B' in model_name_str or 'Llama-3.1-70B' in model_name_str or 'llama70b' in model_name_str:
        model = 'llama70b'
    elif 'QwQ_32B' in model_name_str or 'qwq32b' in model_name_str:
        model = 'qwq32b'
    elif 'Qwen2_5_32B' in model_name_str or 'qwen25_32b' in model_name_str or 'Qwen2.5-32B' in model_name_str:
        model = 'qwen25_32b'
    elif 'Qwen2' in model_name_str: # Broader catch for Qwen2
        model = 'qwen2' 
    else:
        # Simplify common patterns if they exist
        model = model_name_str.replace("meta-llama/", "").replace("meta_llama/", "").lower()
        if model.endswith(".csv"): # Ensure .csv is stripped if it's part of a simple name
            model = model[:-4]
            
    return method, model

def load_all_samples(source_dir_name, 
                     required_source_cols=['claim', 'article_text', 'date'], 
                     rename_date_col_to='article_date',
                     deduplicate_on=['claim', 'article_text'],
                     exclude_files_containing=None,
                     filter_language_code=None): # New param for "sample_ids"
    """
    Load and combine all sample CSV files from a directory.
    Extracts method and model, adds group, handles date renaming, and deduplicates.
    """
    sample_files_pattern = os.path.join(source_dir_name, "*.csv")
    all_csv_files = glob.glob(sample_files_pattern)

    if exclude_files_containing:
        if not isinstance(exclude_files_containing, list):
            exclude_files_containing = [exclude_files_containing]
        
        filtered_files = []
        for f_path in all_csv_files:
            allow_file = True
            for exclusion_keyword in exclude_files_containing:
                if exclusion_keyword in os.path.basename(f_path):
                    allow_file = False
                    break
            if allow_file:
                filtered_files.append(f_path)
        sample_files = filtered_files
        print(f"Excluded files containing: {exclude_files_containing}. Kept {len(sample_files)} out of {len(all_csv_files)} files.")
    else:
        sample_files = all_csv_files

    if not sample_files:
        raise ValueError(f"No sample files found in {source_dir_name} (after exclusions).")

    dfs = []
    for file_path in sample_files:
        try:
            method, model = get_method_and_model(file_path) # Uses the util version
            df = pd.read_csv(file_path)
            df['method'] = method
            df['model'] = model
            df['group'] = f"{method}_{model}"
            df['source_file'] = os.path.basename(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}")
            continue

    if not dfs:
        raise ValueError(f"No data successfully loaded from files in {source_dir_name}.")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total claims from {len(dfs)} files. Groups found: {sorted(combined_df['group'].unique())}")

    # Ensure required source columns exist
    if required_source_cols:
        missing_cols = [col for col in required_source_cols if col not in combined_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required source columns: {missing_cols}. Available: {combined_df.columns.tolist()}")
        
        # Enforce non-missing values for these required columns
        initial_count_before_dropna = len(combined_df)
        combined_df.dropna(subset=required_source_cols, inplace=True)
        if len(combined_df) < initial_count_before_dropna:
            print(f"Removed {initial_count_before_dropna - len(combined_df)} rows due to missing values in required columns: {required_source_cols}.")

    # Filter by language if specified
    if filter_language_code:
        if 'language' in combined_df.columns:
            initial_count_before_lang_filter = len(combined_df)
            combined_df = combined_df[combined_df['language'].astype(str).str.lower() == filter_language_code.lower()]
            if len(combined_df) < initial_count_before_lang_filter:
                print(f"Filtered by language code '{filter_language_code}'. Kept {len(combined_df)} from {initial_count_before_lang_filter} rows.")
            elif initial_count_before_lang_filter > 0 : 
                print(f"Language filter '{filter_language_code}' applied. All {initial_count_before_lang_filter} rows matched or no non-matching rows to remove.")
            else:
                print(f"Language filter '{filter_language_code}' applied, but DataFrame was already empty.")
        else:
            print(f"Warning: Language filter '{filter_language_code}' requested, but 'language' column not found in data. Skipping language filter.")

    # Rename 'date' to 'article_date' (or other specified column) for consistency
    if rename_date_col_to and 'date' in combined_df.columns and 'date' in required_source_cols:
        combined_df = combined_df.rename(columns={'date': rename_date_col_to})
        print(f"Renamed source column 'date' to '{rename_date_col_to}'.")

    # Deduplicate claims if specified
    if deduplicate_on and all(col in combined_df.columns for col in deduplicate_on):
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=deduplicate_on, keep='first')
        if len(combined_df) < initial_count:
            print(f"Removed {initial_count - len(combined_df)} duplicate claims (based on {deduplicate_on}).")
    elif deduplicate_on:
        print(f"Warning: Could not deduplicate. One or more columns for deduplication {deduplicate_on} not found in DataFrame.")

    return combined_df 