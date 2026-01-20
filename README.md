# NBA Player Trajectory Prediction

**Course:** CS4641 Machine Learning @ Georgia Tech  
**Project Website:** [Team Page](https://github.gatech.edu/pages/xliang304/CS4641-Project-Team/)  
**Dataset:** [Google Sheets Link](https://docs.google.com/spreadsheets/d/1-fuobSC5Hv7Sgvg_6cW7tAjUuDcMnbsHq7pEwBeC2Wg/edit?usp=sharing)

---

## 📂 Repository Structure

### Data
Located in the `/datasets` folder.

| File | Description |
| :--- | :--- |
| `NBA ML Dataset - Sheet1.csv` | Original raw dataset. |
| `nba_players_cleaned.csv` | Dataset after preprocessing. |
| `nba_linreg_train.csv` | Training subset for Linear Regression. |
| `nba_linreg_test.csv` | Testing subset for Linear Regression. |

### Source Code
Run these files to train models or generate graphs.

* **`data_preprocessing.py`**: Methods used to clean and prepare the dataset.
* **`data_visualization.py`**: Generates graphs to analyze feature relationships.
* **`models_visualization.py`**: Functions for visualizing model results.
* **`linreg_model.py`**: Linear Regression model implementation.
* **`decisiontree_model.py`**: Decision Tree model implementation.
* **`knn_model.py`**: KNN Regression model implementation.

### Visualizations
Located in the `/visualization` folder.

#### Linear Regression
* `linreg_regplane_all-stars_by_y4_fixed.png`: 3D regression plane (All-Stars by Y4 fixed).
* `linreg_regplane_y1_ppg_fixed.png`: 3D regression plane (Y1 PPG fixed).
* `linreg_regplane_y3_ppg_fixed.png`: 3D regression plane (Y3 PPG fixed).
* `linreg_residualbar.png` / `linreg_residualplot.png`: Residual analysis (Histogram/Scatterplot).
* `linreg_scatterplot.png`: Predicted vs. Actual All-Stars.

#### Decision Tree
* `dtreg_tree`: Visualization of the decision tree structure.
* `dtreg_scatterplot_y3_bpg.png`: Predicted vs. Actual plotted against Y3 BPG.
* `dtreg_scatterplot_y3_ppg.png`: Predicted vs. Actual plotted against Y3 PPG.
* `dtreg_scatterplot_y4_gp.png`: Predicted vs. Actual plotted against Y4 Games Played.
* `dtreg_scatterplot.png`: General Predicted vs. Actual All-Stars.
* `dtreg_residualbar.png` / `dtreg_residualplot.png`: Residual analysis.

#### KNN Regression
* `knnreg_scatterplot.png`: Predicted vs. Actual All-Stars.
* `knnreg_residualbar.png` / `knnreg_residualplot.png`: Residual analysis.

#### Performance Comparison
* `mse_piechart.png`: Model with best MSE across all test sizes and top 9 features.
* `r2_piechart.png`: Model with best R² across all test sizes and top 9 features.
