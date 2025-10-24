import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def one_hot_encoding_positions(df):
    position_map = {
    'PG': ['pg'], 'PG/SG': ['pg', 'sg'],
    'SG': ['sg'], 'SG/SF': ['sg', 'sf'], 'G/F': ['sg', 'sf'],
    'SF': ['sf'], 'SF/PF': ['sf', 'pf'], 'G': ['pg', 'sg'], 'F': ['sf', 'pf'],
    'PF': ['pf'], 'PF/C': ['pf', 'c'], 'F/C': ['pf', 'c'],
    'C': ['c']
    }

    for pos in ['pg', 'sg', 'sf', 'pf', 'c']:
        df[f'is_{pos}'] = 0
    
    for index, row in df.iterrows():
        positions = position_map.get(row['position'], [])
        for pos in positions:
            df.at[index, f'is_{pos}'] = 1
    
    df.drop(columns=['position'], inplace=True)
    position_cols = ['is_pg', 'is_sg', 'is_sf', 'is_pf', 'is_c']
    other_cols = [col for col in df.columns if col not in position_cols]
    df = df[position_cols + other_cols]

    return df

def min_max_scaling(df):
    exclude_cols = ['name', 'position', 'active']
    numeric_cols = df.columns.difference(exclude_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    percentage_cols = [
    'y1_fg%', 'y1_3p%', 'y1_ft%', 'y2_fg%', 'y2_3p%', 'y2_ft%',
    'y3_fg%', 'y3_3p%', 'y3_ft%', 'y4_fg%', 'y4_3p%', 'y4_ft%'
    ]
    features_to_normalize = numeric_cols.difference(percentage_cols + ['all-stars_by_y4', 'total_all-stars'])

    scaler = MinMaxScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    return df

def binary_active(df):
    df['active'] = df['active'].map({'Yes': 1, 'No': 0})
    return df
