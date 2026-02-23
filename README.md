# NBA Player Trajectory Prediction

**Course:** CS4641 Machine Learning @ Georgia Tech  
**Project Website:** [Team Page](https://github.gatech.edu/pages/xliang304/CS4641-Project-Team/)  
**Dataset:** [Google Sheets Link](https://docs.google.com/spreadsheets/d/1-fuobSC5Hv7Sgvg_6cW7tAjUuDcMnbsHq7pEwBeC2Wg/edit?usp=sharing)

---

## Overview

Predicting long-term NBA player success is one of the most consequential — and difficult — decisions a franchise faces. This project uses machine learning to forecast **how many All-Star appearances an NBA player will earn over their career**, based solely on their statistics from their rookie contract (Years 1–4).

### Why This Matters
General managers routinely face high-stakes contract extension decisions with limited signal. Early indicators like draft position and college stats offer some guidance, but teams consistently struggle to accurately project long-term potential. Traditional approaches apply a one-size-fits-all model to player assessment — our project improves on this by exploring **position-based feature weighting** (e.g., assists matter more for guards than centers) and accounting for **evolving league trends** like the rise of three-point shooting and faster pace of play.

### Data
We collected statistics for **249 NBA players** drafted between 2005–2020. For each player, we tracked Years 1–4 games played, games started, MPG, FG%, 3P%, FT%, RPG, APG, SPG, BPG, and PPG — along with their total All-Star appearances by Year 4. The dataset includes all 80 players from that era with at least one All-Star selection, the four legends above, and 165 randomly sampled players.

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
