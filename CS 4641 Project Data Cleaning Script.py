# Data Cleaning !!!
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# 1) Load current dataset (uncleaned)
file_path = "/content/FILE NAME.csv"
df = pd.read_csv(file_path)

# 2) Standardize column names (just make lowercase and remove spaces)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# 3) Convert draft pick to numeric
df['year_drafted'] = pd.to_numeric(df['year_drafted'], errors='coerce')

# 4) Standardize position names
# --> Players that have multiple positions play the position that comes first
position_mapping = {
    'PG': 'Point Guard', 'SG': 'Shooting Guard', 'SF': 'Small Forward',
    'PF': 'Power Forward', 'C': 'Center', 'PG/SG': 'Point Guard',
    'SG/SF': 'Shooting Guard', 'SF/PF': 'Small Forward', 'PF/C': 'Power Forward',
    'G/F': 'Small Forward', 'F': 'Small Forward', 'F/C' : 'Power Forward'
}
df['position'] = df['position'].map(lambda x: position_mapping.get(x, x))

# 5) Convert all statistics columns to numeric
exclude_cols = ['name', 'position', 'active']
numeric_cols = df.columns.difference(exclude_cols)
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# ---------------------------------------------------------------------------------------
# 6) Exclude percentage-based stats from normalization
percentage_cols = [
    'y1_fg%', 'y1_3p%', 'y1_ft%', 'y2_fg%', 'y2_3p%', 'y2_ft%',
    'y3_fg%', 'y3_3p%', 'y3_ft%', 'y4_fg%', 'y4_3p%', 'y4_ft%'
]

features_to_normalize = numeric_cols.difference(percentage_cols + ['all-stars_by_y4', 'total_all-stars'])

# 7) IMPORTANT: Normalize statistics using Min-Max Scaling
# --> Used to scale the data between 0 and 1
# --> High draft picks will be closer to 1, low draft picks will be closer to 0

# Normalize only the feature columns
scaler = MinMaxScaler()
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# 8) Save cleaned dataset
cleaned_file_path = "/content/nba_players_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to: {cleaned_file_path}")
