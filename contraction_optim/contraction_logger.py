import pandas as pd
import pickle
import os
import sys

SEQUENTIAL = False

def log_arguments(*args):
    args = list(args)
    if len(args) != 11:
        raise ValueError("Exactly 11 arguments are required")
    
    for i in range (2, 5):
        args[i] = list(args[i])
    
    args = [str(i) for i in args]
    
    print(args)

    log_file = 'logged_dimensions_tensorized_nn.pkl'
    
    # Load existing dataframe from pickle if it exists
    if os.path.exists(log_file):
        with open(log_file, 'rb') as f:
            df = pickle.load(f)
    else:
        if SEQUENTIAL:
            df = pd.DataFrame(columns=["original_A", "original_B", "A", "B", "C", "contraction_dim", "type", "number", "total_time", "A_name", "B_name", "C_name"])
        else:
            df = pd.DataFrame(columns=["A", "B", "C", "contraction_dim", "type", "number", "total_time"])

    if SEQUENTIAL:
        # Append new arguments to the dataframe
        new_entry = pd.DataFrame([args + (1,)], columns=df.columns.tolist())
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        # Append new arguments to the dataframe
        # Check if there are existing entries with the same values in A, B, and C columns
        existing_entry = df[(df['A'] == args[2]) & (df['B'] == args[3]) & (df['C'] == args[4])]

        if not existing_entry.empty:
            # If such entries exist, increase their number in the 'number' column
            df.loc[existing_entry.index, 'number'] += 1
            df.loc[existing_entry.index, 'total_time'] += args[9]
        else:
            # If no such entries exist, add a new entry with 'number' set to 1
            new_entry = pd.DataFrame([args[2:8] + [1,0]], columns=df.columns.tolist())
            df = pd.concat([df, new_entry], ignore_index=True)

    # Save the updated dataframe back to pickle
    with open(log_file, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    log_arguments(*sys.argv[1:])