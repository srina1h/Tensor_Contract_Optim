import pandas as pd
import pickle
import os
import sys

def log_arguments(*args):
    if len(args) != 7:
        raise ValueError("Exactly 7 arguments are required")

    log_file = 'logged_dimensions_tensorized_nn.pkl'
    
    # Load existing dataframe from pickle if it exists
    if os.path.exists(log_file):
        with open(log_file, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame(columns=["original_A", "original_B", "A", "B", "C", "contraction_dim", "type", "number"])
    
    # Append new arguments to the dataframe
    # Check if there are existing entries with the same values in A, B, and C columns
    existing_entry = df[(df['A'] == args[2]) & (df['B'] == args[3]) & (df['C'] == args[4])]

    if not existing_entry.empty:
        # If such entries exist, increase their number in the 'number' column
        df.loc[existing_entry.index, 'number'] += 1
    else:
        # If no such entries exist, add a new entry with 'number' set to 1
        new_entry = pd.DataFrame([args + (1,)], columns=df.columns.tolist())
        df = pd.concat([df, new_entry], ignore_index=True)
    
    # Save the updated dataframe back to pickle
    with open(log_file, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    log_arguments(*sys.argv[1:])