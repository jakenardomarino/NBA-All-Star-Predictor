# NBA Player Trajectory Prediction
[Our Project Website](https://github.gatech.edu/pages/xliang304/CS4641-Project-Team/)

Dataset link: https://docs.google.com/spreadsheets/d/1-fuobSC5Hv7Sgvg_6cW7tAjUuDcMnbsHq7pEwBeC2Wg/edit?usp=sharing

/datasets: folder with the original and cleaned datasets

/datasets/NBA ML Dataset - Sheet1.csv: original dataset csv file

/datasets/nba_players_cleaned.csv: dataset after preprocessing is complete

/datasets/nba_linreg_train.csv: subsection of the preprocessed dataset used in training the linear regression model

/datasets/nba_linreg_test.csv: subsection of the preprocessed dataset used in testing the linear regression model


data_preprocessing.py: file containing the preprocessing methods used to clean the dataset

data_visualization.py: file containing functions for looking at the relationships between features (run to see graphs)

linreg_model.py: file containing the linear regression model (run to see model, results)

decisiontree_model.py: file containing the decision tree model (run to see model, results)

knn_model.py: file containing the KNN regression model (run to see model, results)

models_visualization.py: file containing functions for looking at the visualization results of each model


/visualization: folder with all the visualizations (graphs, etc.) of the models

/visualization/linreg_regplane_all-stars_by_y4_fixed.png: 3D regression plane with All-Stars by Y4 feature fixed

/visualization/linreg_regplane_y1_ppg_fixed.png: 3D regression plane with Y1 PPG feature fixed

/visualization/linreg_regplane_y3_ppg_fixed.png: 3D regression plane with Y3 PPG feature fixed

/visualization/linreg_residualbar.png: Histogram of residuals for the linear regression model

/visualization/linreg_residualplot.png: Scatterplot of residuals for the linear regression model

/visualization/linreg_scatterplot.png: Scatterplot of predicted vs. actual all-stars for the linear regression model

/visualization/dtreg_scatterplot_y3_bpg.png: Scatterplot of predicted vs. actual all stars plotted against Y3 BPG

/visualization/dtreg_scatterplot_y3_ppg.png: Scatterplot of predicted vs. actual all stars plotted against Y3 PPG

/visualization/dtreg_scatterplot_y4_gp.png: Scatterplot of predicted vs. actual all stars plotted against Y4 games played

/visualization/dtreg_scatterplot.png: Scatterplot of predicted vs. actual all stars for the decision tree model

/visualization/dtreg_tree: Visualization of the decision tree

/visualization/dtreg_residualbar.png: Histogram of residuals for the decision tree model

/visualization/dtreg_residualplot.png: Scatterplot of residuals for the decision tree model

/visualization/knnreg_residualbar.png: Histogram of residuals for the KNN regression model

/visualization/knnreg_residualplot.png: Scatterplot of residuals for the KNN regression model

/visualization/knnreg_scatterplot.png: Scatterplot of predicted vs. actual all stars for the KNN regression model

/visualization/mse_piechart.png: Pie chart of the model with the best MSE value over all test sizes (0.05, 0.06, ..., 0.20) and subset of top 9 correlated features

/visualization/r2_piechart.png: Pie chart of the model with the best R2 value over all test sizes (0.05, 0.06, ..., 0.20) and subset of top 9 correlated features