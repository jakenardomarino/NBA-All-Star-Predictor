---
layout: default
title: Project Proposal
---
[< Home](./index.md)

# Proposal Report

### Introduction/Background

The topic that our group will pursue is predicting how many All-Stars an NBA player will obtain over the course of their career, given their statistics during their rookie contract. This could be used by NBA general managers when negotiating contract extensions with NBA players. Most literature in this field predicts the success of NBA players given their college careers through regression analysis. While these papers did not use machine learning, they are useful for feature selection. Potential features include position, draft pick number, minutes played, FT%, assists, steals, blocks, turnovers, personal fouls, and points [<a href="#ref1">1</a>]. Different features could be used depending on whether a player is a guard, forward, or center [<a href="#ref2">2</a>]. The potential datasets that we are considering are RapidAPI’s NBA dataset and SportRadar’s NBA API. Both datasets contain player career information for thousands of players, which we can separate by year to look at their rookie contract statistics.

### Problem Definition

The problem we intend to solve with ML is that many NBA teams struggle and continue to face critical decisions when deciding to build their franchise around their young players. Moreover, while early statistics such as college performance or draft position provide some sort of indication of overall potential, teams continue to struggle with accurately forecasting long term success. Unlike previous literature that have applied a one size fits all approach to player assessment, we intend to tailor predictions through position-based feature weighting (ie. assists would be more impactful for a guard) as well as analyze any league-wide trends such as the increased emphasis of three-point shooting and pace of the game. Thus, our model can be represented as much more dynamic and adaptable in comparison to previous static correlations that have been used.

### Methods

Data preprocessing is crucial for machine learning. We’ve chosen methods like PCA for dimension reduction to eliminate redundancies, KNN imputation for missing values to preserve data integrity, and Z-score and Min-Max scaling for normalization to standardize feature distribution. For supervised tasks, we’ll use regression and classification models such as Random Forest and XGBoost, while unsupervised learning will employ clustering algorithms like K-Means, Hierarchical Clustering, and DBSCAN. These processing strategies and ML methods were considered to enhance model accuracy and improve computational efficiency.

### Results and Discussion

To evaluate our model's effectiveness in predicting an NBA player’s All-Star appearances, we will use the coefficient of determination (R²), F1-score, and mean squared error (MSE). Our overarching goal to ensure the model correctly predicts future All-Star appearances of players based on their rookie contracts. Using R² would allow us to measure how well the model’s predictions match actual outcomes. F1-score balances precision and recall, making it useful in detecting cases of false positives (wrongly predicting a player will have a certain range of All-Stars) and false negatives (missing a future All-Star). It has also been used as a metric to evaluate ML models in other papers whose goal was to predict NBA players’ performance and popularity [<a href="#ref3">3</a>]. MSE helps quantify the error in continuous predictions, such as the exact number of All-Star selections a player will obtain. For our models, success will be determined by achieving high predictive accuracy while minimizing misclassifications, ensuring NBA teams gain actionable insights for contract decisions. Our goal is to create an accurate, environmentally sustainable, and specialized model that can be used to predict the careers of the next generation of NBA superstars. Both GMs and agents can use our data while negotiating multi-million dollar contracts. We are excited to see how machine learning models like ours can impact the NBA landscape.

### Video Presentation
Check out our video presentation on [YouTube](https://youtu.be/JPbk2N-Nv8g)

### Gantt Chart
The Gantt Chart can be accessed [here](https://docs.google.com/spreadsheets/d/10yeWSocKFXN5sgRKlP1bHP6SGMhg_Mfy/edit?usp=sharing&ouid=112407887011389750537&rtpof=true&sd=true)

### Contribution Table



| Name   | Proposal Contributions                                       |
| ------ | ------------------------------------------------------------ |
| Aarav  | Wrote introduction and background. Recorded video presentation |
| Aryan  | Wrote results and discussion                                 |
| Ekechi | Wrote methods                                                |
| Jake   | Wrote problem definition                                     |
| Xinyu  | Set up Github Pages, drew Gantt Chart, and wrote contribution table |



### References

<div style="padding-left: 2em; text-indent: -2em;">
<p id="ref1">[1] B. Oguntimein and D. Coates, “The Length and Success of NBA Careers: Does College Production Predict Professional Outcomes?,” <i>International Journal of Sport Finance</i>, vol. 5, no. 1, pp. 4–26, 2010.</p>

<p id="ref2">[2] W. Abrams, J. C. Barnes, and A. Clement, “Relationship of selected pre-NBA career variables to NBA players’ career longevity,” <i>The Sport Journal</i>, vol. 11, no. 2, 2008.</p>

<p id="ref3">[3] N. H. Nguyen, D. T. A. Nguyen, B. Ma, and J. Hu, “The application of machine learning and deep learning in sport: predicting NBA players’ performance and popularity,” <i>Journal of Information and Telecommunication</i>, pp. 1–19, Sep. 2021.</p>
</div>
