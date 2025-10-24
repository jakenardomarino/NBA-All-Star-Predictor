import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import one_hot_encoding_positions, min_max_scaling, binary_active
from models_visualization import dt_scatterplot, dt_plot_tree, pred_vs_actual, residual_plots

file_path = "datasets/NBA ML Dataset - Sheet1.csv"
original_df = pd.read_csv(file_path, index_col=False)

# Important variable declaration -- changed to optimize model results
test_size = 0.07
feature_list = ['y3_bpg', 'y3_ppg', 'y4_gp']
max_depth=3

# Data preprocessing section
df = original_df.copy(deep=True)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
df.drop(columns=['name', 'year_drafted'], inplace=True, errors='ignore')
df = one_hot_encoding_positions(df)
df = binary_active(df)
df = min_max_scaling(df)
df.to_csv("datasets/nba_players_cleaned.csv", index=False)

# Splitting data into train and test sets, each with proportional number of non-allstars
players_with_allstars = df[df['total_all-stars'] > 0]
players_without_allstars = df[df['total_all-stars'] == 0]

train_with_allstars, test_with_allstars = train_test_split(players_with_allstars, test_size=test_size, random_state=42)
train_without_allstars, test_without_allstars = train_test_split(players_without_allstars, test_size=test_size, random_state=42)
final_train_set = pd.concat([train_with_allstars, train_without_allstars]).sample(frac=1, random_state=42)
final_test_set = pd.concat([test_with_allstars, test_without_allstars]).sample(frac=1, random_state=42)

train_file_path = "datasets/nba_train.csv"
test_file_path = "datasets/nba_test.csv"
final_train_set.to_csv(train_file_path, index=False)
final_test_set.to_csv(test_file_path, index=False)

X_train = final_train_set.drop(columns=['total_all-stars', 'active'])
y_train = final_train_set['total_all-stars']
X_test = final_test_set.drop(columns=['total_all-stars', 'active'])
y_test = final_test_set['total_all-stars']

# Decision Tree model
dtreg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
dtreg.fit(X_train[feature_list], y_train)
y_pred = dtreg.predict(X_test[feature_list])

# Decision Tree model metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# Decision Tree visualization
pred_vs_actual("dtreg", y_pred, y_test)
residual_plots("dtreg", y_pred, y_test)
dt_scatterplot(X_test, y_test, y_pred, features=feature_list)
dt_plot_tree(dtreg=dtreg, features=feature_list, depth=max_depth)