import pandas as pd
import matplotlib.pyplot as plt

# Dataset Visualization Methods

def position_distribution(df):
    positions = ["is_pg", "is_sg", "is_sf", "is_pf", "is_c"]
    position_counts = df[positions].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(positions, position_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Position")
    plt.ylabel("Number of Players")
    plt.title("Distribution of Players by Position")
    plt.show()

def all_stars_by_position(df):
    positions = ["is_pg", "is_sg", "is_sf", "is_pf", "is_c"]
    position_labels = ["PG", "SG", "SF", "PF", "C"]
    all_star_data = [df[df[pos] == 1]['total_all-stars'] for pos in positions]

    plt.figure(figsize=(10, 6))
    plt.boxplot(all_star_data, labels=position_labels, patch_artist=True)
    plt.xlabel("Position")
    plt.ylabel("Total All-Stars")
    plt.title("Total All-Star Selections by Position")
    plt.show()

def feat_correlation_heatmap(df):
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')

    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    fig.colorbar(cax)
    plt.title("Feature Correlation Heatmap", pad=20)
    plt.show()

    correlation_matrix = df.corr()
    correlation_with_all_stars = correlation_matrix["total_all-stars"].dropna()
    top_10_features = correlation_with_all_stars.abs().sort_values(ascending=False).head(10)
    return top_10_features

def draft_rnd_scatterplot(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['pick_round'], df['total_all-stars'], alpha=0.5)
    plt.xlabel("Draft Round")
    plt.ylabel("Total All Stars")
    plt.title("Draft Round vs. All-Star Selections")
    plt.show()

def y4_ppg_scatterplot(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['y4_ppg'], df['total_all-stars'], alpha=0.5, edgecolors='black')
    plt.xlabel("Year 4 PPG")
    plt.ylabel("Total All-Stars")
    plt.title("Year 4 PPG vs. Total All-Stars")
    plt.show()

def all_stars_y4_scatterplot(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['all-stars_by_y4'], df['total_all-stars'], alpha=0.5, edgecolors='black')
    plt.xlabel("All Stars by Year 4")
    plt.ylabel("Total All-Stars")
    plt.title("All Stars by Year 4 vs. Total All-Stars")
    plt.show()

def ppg_per_year(df):
    plt.figure(figsize=(8, 6))
    y1_avg = df['y1_ppg'].mean()
    y2_avg = df['y2_ppg'].mean()
    y3_avg = df['y3_ppg'].mean()
    y4_avg = df['y4_ppg'].mean()

    plt.plot([1, 2, 3, 4], [y1_avg, y2_avg, y3_avg, y4_avg], marker="o", linestyle="-", color='b', label="PPG")
    plt.xlabel("Years in NBA")
    plt.ylabel("Average PPG")
    plt.title("Average PPG Over 4 Years")
    plt.xticks([1, 2, 3, 4], labels=["Year 1", "Year 2", "Year 3", "Year 4"])
    plt.legend()
    plt.show()

def ppg_allstar_y4(df):
    plt.scatter(df['y1_ppg'], df['y3_ppg'], c=df['total_all-stars'], cmap='coolwarm', s=df['all-stars_by_y4']*10)
    plt.colorbar(label="Total All-Stars")
    plt.xlabel("Y1 PPG")
    plt.ylabel("Y3 PPG")
    plt.title("3D Data Projection with All-Stars by Y4 as Marker Size")
    plt.show()

file_path = "nba_players_cleaned.csv"
df = pd.read_csv(file_path, index_col=False)
position_distribution(df)
all_stars_by_position(df)
feat_correlation_heatmap(df)
draft_rnd_scatterplot(df)
y4_ppg_scatterplot(df)
all_stars_y4_scatterplot(df)
ppg_per_year(df)
ppg_allstar_y4(df)