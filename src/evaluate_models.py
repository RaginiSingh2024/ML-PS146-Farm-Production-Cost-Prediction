"""
src/evaluate_models.py
Evaluation metrics calculation and visualization utilities for regression models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set clean aesthetic styling for university presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def calculate_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Calculate MAE, MSE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': round(float(mae), 2),
        'MSE': round(float(mse), 2),
        'RMSE': round(float(rmse), 2),
        'R²': round(float(r2), 4)
    }

def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str = "results/model_comparison.png"):
    """
    Generate side-by-side comparison charts for RMSE and R² across all models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. RMSE Comparison (Lower is better)
    sns.barplot(
        data=comparison_df,
        x='Model',
        y='RMSE',
        hue='Model',
        palette='crest',
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title('Model Comparison — Root Mean Squared Error (RMSE)\n(Lower is Better)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel('RMSE (₹)', fontsize=11)
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=25)
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f'₹{height:,.0f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points')

    # 2. R² Comparison (Higher is better)
    sns.barplot(
        data=comparison_df,
        x='Model',
        y='R²',
        hue='Model',
        palette='viridis',
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Model Comparison — Coefficient of Determination ($R^2$)\n(Higher is Better)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_ylabel('$R^2$ Score', fontsize=11)
    axes[1].set_xlabel('')
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=25)
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f'{height:.4f}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved model comparison chart to '{output_path}'.")

def plot_actual_vs_predicted(y_true, y_pred, model_name: str, output_path: str = "results/actual_vs_predicted.png"):
    """
    Plot actual vs predicted scatter plot with 45-degree ideal fit line.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.5, label='Farms (Test Set)')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Ideal 1:1 Prediction Line')
    
    plt.title(f'Actual vs. Predicted Farm Production Cost\nBest Model: {model_name}', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Actual Total Production Cost (₹)', fontsize=11)
    plt.ylabel('Predicted Total Production Cost (₹)', fontsize=11)
    plt.gca().xaxis.set_major_formatter('₹{x:,.0f}')
    plt.gca().yaxis.set_major_formatter('₹{x:,.0f}')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved actual vs predicted plot to '{output_path}'.")

def plot_residual_distribution(y_true, y_pred, model_name: str, output_path: str = "results/residual_plot.png"):
    """
    Plot residual scatter plot and residual distribution histogram.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='#2ca02c', edgecolors='k', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title(f'Residuals vs. Predicted Values ({model_name})', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel('Predicted Production Cost (₹)', fontsize=11)
    axes[0].set_ylabel('Residuals (Actual - Predicted) (₹)', fontsize=11)
    axes[0].yaxis.set_major_formatter('₹{x:,.0f}')
    axes[0].xaxis.set_major_formatter('₹{x:,.0f}')

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, color='#2ca02c', ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residual Error Distribution (Normality Check)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel('Residual Error (₹)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].xaxis.set_major_formatter('₹{x:,.0f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved residual plot to '{output_path}'.")

def plot_feature_importance(importance_df: pd.DataFrame, output_path: str = "results/feature_importance.png"):
    """
    Plot horizontal bar chart of feature importances.
    """
    plt.figure(figsize=(10, 6))
    sorted_df = importance_df.sort_values(by='Importance', ascending=True)
    colors = sns.color_palette("mako", len(sorted_df))
    bars = plt.barh(sorted_df['Feature'], sorted_df['Importance'], color=colors, edgecolor='k', linewidth=0.5)
    
    plt.title('Feature Importance / Influence on Farm Production Cost', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Normalized Importance Score (Relative Contribution)', fontsize=11)
    plt.ylabel('Production Factors & Input Variables', fontsize=11)
    
    for bar in bars:
        width = bar.get_width()
        plt.annotate(f'{width:.3f} ({width*100:.1f}%)',
                     (width, bar.get_y() + bar.get_height() / 2.),
                     ha='left', va='center', fontsize=9, xytext=(5, 0),
                     textcoords='offset points')
                     
    plt.xlim(0, max(sorted_df['Importance']) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved feature importance chart to '{output_path}'.")

def plot_farm_area_vs_cost(df: pd.DataFrame, output_path: str = "results/farm_area_vs_cost.png"):
    """
    Analyze and plot the relationship between Farm_Area and Total_Production_Cost across crop types.
    """
    plt.figure(figsize=(9, 6))
    palette = {'Wheat': '#e67e22', 'Rice': '#2980b9', 'Cotton': '#8e44ad', 'Sugarcane': '#27ae60', 'Maize': '#f39c12'}
    
    sns.scatterplot(
        data=df,
        x='Farm_Area',
        y='Total_Production_Cost',
        hue='Crop_Type',
        palette=palette,
        alpha=0.75,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Add overall trend line
    sns.regplot(
        data=df,
        x='Farm_Area',
        y='Total_Production_Cost',
        scatter=False,
        color='#333333',
        line_kws={'linestyle': '--', 'linewidth': 2, 'label': 'Overall Cost Trend'}
    )
    
    plt.title('Farm Size (Area in Acres) vs. Total Production Cost', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Farm Area (Acres)', fontsize=11)
    plt.ylabel('Total Production Cost (₹)', fontsize=11)
    plt.gca().yaxis.set_major_formatter('₹{x:,.0f}')
    plt.legend(title='Crop Type', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Farm Area vs Cost chart to '{output_path}'.")
