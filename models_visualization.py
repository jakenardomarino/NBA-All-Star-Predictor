import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def pred_vs_actual(model_name, y_pred, y_test):
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=2)
    plt.xlabel("True # All‑Stars")
    plt.ylabel("Predicted # All‑Stars")
    plt.title("Predicted vs. Actual")
    plt.savefig('visualization/' + model_name + '_scatterplot.png')
    plt.show()

def residual_plots(model_name, y_pred, y_test):
    residuals = y_pred - y_test
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(0, y_pred.min(), y_pred.max(), 'r', linestyles='dashed')
    plt.xlabel("Predicted # All‑Stars")
    plt.ylabel("Residual (Pred – True)")
    plt.title("Residuals vs. Predicted")
    plt.savefig('visualization/' + model_name + '_residualplot.png')
    plt.show()

    plt.figure(figsize=(5,3))
    plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Error = Predicted – True")
    plt.ylabel("Count")
    plt.title("Histogram of Residuals")
    plt.savefig('visualization/' + model_name + '_residualbar.png')
    plt.show()

def regression_plane(model, X_train, y_pred, feature_normalized):
    if feature_normalized == 'all-stars_by_y4':
        x1 = 'y1_ppg'
        x2 = 'y3_ppg'
    elif feature_normalized == 'y1_ppg':
        x1 = 'all-stars_by_y4'
        x2 = 'y3_ppg'
    elif feature_normalized == 'y3_ppg':
        x1 = 'all-stars_by_y4'
        x2 = 'y1_ppg'
    else:
        return None
    
    X1 = X_train[x1]
    X2 = X_train[x2]
    X3 = X_train[feature_normalized]
    X3_fixed = np.median(X3)
    x1_range = np.linspace(X1.min(), X1.max(), 50)
    x2_range = np.linspace(X2.min(), X2.max(), 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    beta0 = model.intercept_
    beta1, beta2, beta3 = model.coef_
    Y_grid = beta0 + beta1 * X1_grid + beta2 * X2_grid + beta3 * X3_fixed
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1, X2, y_pred, c=X3, cmap="coolwarm", edgecolors="k", label="Predictions")
    ax.plot_surface(X1_grid, X2_grid, Y_grid, color="cyan", alpha=0.5)

    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_zlabel("total_all-stars")
    ax.set_title("Regression Plane (Fixing " + feature_normalized + ")")
    plt.savefig('visualization/linreg_regplane_' + feature_normalized + '_fixed.png')
    plt.show()

def dt_scatterplot(X_test, y_test, y_pred, features):
    for feature in features:
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test[feature], y_test, color='blue', label='Actual', alpha=0.6)
        plt.scatter(X_test[feature], y_pred, color='green', label='Predicted', alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("Total All-Stars")
        plt.title(f"Decision Tree Prediction vs Actual ({feature})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualization/dtreg_scatterplot_' + feature + '.png')
        plt.show()

def dt_plot_tree(dtreg, features, depth):
    plt.figure(figsize=(20, 10))
    plot_tree(
        dtreg,
        feature_names=features,
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=3
    )
    plt.title("Decision Tree Visualization (max_depth=" + str(depth) + ")")
    plt.savefig('visualization/dtreg_tree.png')
    plt.show()