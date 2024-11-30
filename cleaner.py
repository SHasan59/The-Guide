import pandas as pd

# File paths
data = "/Users/estebanm/Desktop/carShopping_tool/Final_Cleaned_Dataset.csv"
output_path = "/Users/estebanm/Desktop/carShopping_tool/Final_Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame

data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
missing_years = data['Year'].isnull().sum()
print(f"Number of missing or invalid 'Year' entries: {missing_years}")

