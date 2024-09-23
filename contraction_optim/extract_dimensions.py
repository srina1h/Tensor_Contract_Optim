import pandas as pd
import pickle

# Path to the pickle file
pickle_file_path = '/home/ssrini27/Tensor_Contract_Optim/logged_dimensions_tensorized_nn.pkl'

# Load the pickle file into a DataFrame
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)

# Export the DataFrame to an xlsx file
xlsx_file_path = 'a100_combined_cupy.xlsx'
df.to_excel(xlsx_file_path, index=False)

print(f"Data has been successfully exported to {xlsx_file_path}")