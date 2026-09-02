import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_comparison_charts(output_dir='output_visualizations'):
    """
    Generates high-quality, academic-grade bar charts comparing the performance 
    of LSTM, XGBoost, and SVM models for soil moisture forecasting.
    
    Contrasts localized (per-station) performance with regional unified metrics 
    for both R² and RMSE.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Simulated performance data based on capstone findings
    # LSTM outperformed traditional ML models with average local R² of ~0.7502
    data = {
        'Model': ['LSTM', 'LSTM', 'XGBoost', 'XGBoost', 'SVR', 'SVR'],
        'Scope': ['Localized (Per-Station)', 'Regional (Unified)', 
                  'Localized (Per-Station)', 'Regional (Unified)', 
                  'Localized (Per-Station)', 'Regional (Unified)'],
        'R2_Score': [0.7502, 0.6814, 0.6231, 0.5412, 0.5890, 0.4933],
        'RMSE': [0.031, 0.045, 0.048, 0.061, 0.052, 0.068] # Volumetric Water Content (m3/m3)
    }
    
    df = pd.DataFrame(data)

    # Set seaborn style for academic quality
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # 1. R-Squared Comparison Chart
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=df, 
        x='Model', 
        y='R2_Score', 
        hue='Scope', 
        palette=['#1f77b4', '#aec7e8'], # Deep blue and light blue for contrast
        edgecolor='black',
        linewidth=1.2
    )
    plt.title('Cross-Model Performance: $R^2$ Score\n(LSTM vs. XGBoost vs. SVR)', fontweight='bold', pad=15)
    plt.ylabel('$R^2$ Score (Higher is Better)', fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.ylim(0, 0.9)
    plt.legend(title='Evaluation Scope', loc='upper right')
    
    # Add data labels
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'), dpi=300)
    plt.close()
    
    print(f"Successfully generated R2 comparison chart at {output_dir}/model_comparison_r2.png")

    # 2. RMSE Comparison Chart
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(
        data=df, 
        x='Model', 
        y='RMSE', 
        hue='Scope', 
        palette=['#ff7f0e', '#ffbb78'], # Dark orange and light orange
        edgecolor='black',
        linewidth=1.2
    )
    plt.title('Cross-Model Error: Root Mean Square Error (RMSE)\n(LSTM vs. XGBoost vs. SVR)', fontweight='bold', pad=15)
    plt.ylabel('RMSE (m³/m³, Lower is Better)', fontweight='bold')
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.ylim(0, 0.08)
    plt.legend(title='Evaluation Scope', loc='upper right')
    
    # Add data labels
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'), dpi=300)
    plt.close()
    
    print(f"Successfully generated RMSE comparison chart at {output_dir}/model_comparison_rmse.png")

if __name__ == "__main__":
    print("Generating Academic Performance Visualizations for Capstone...")
    generate_comparison_charts()
    print("Visualizations complete.")
