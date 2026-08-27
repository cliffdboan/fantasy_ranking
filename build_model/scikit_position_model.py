import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
from data_cleaning.create_df import all_data
from model_core import build_lagged_dataset, fit_position_model

# Year whose trained models/visuals are saved under models/{YEAR} and
# model_analysis/{YEAR}. Pass on the command line: python scikit_position_model.py 2026
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

def create_position_visuals(position, year, feature_importance, y_test, y_pred, mae, r2):
    """Create visualizations for position model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{position} Model Analysis', fontsize=16)

    # Feature importance
    top_features = feature_importance.head(10)
    axes[0,0].barh(range(len(top_features)), top_features['importance'])
    axes[0,0].set_yticks(range(len(top_features)))
    axes[0,0].set_yticklabels(top_features['feature'])
    axes[0,0].set_title('Top 10 Feature Importance')
    axes[0,0].invert_yaxis()

    # Prediction vs Actual scatter
    axes[0,1].scatter(y_test, y_pred, alpha=0.6)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0,1].set_xlabel('Actual Fantasy Points')
    axes[0,1].set_ylabel('Predicted Fantasy Points')
    axes[0,1].set_title(f'Predictions vs Actual\nMAE: {mae:.1f}, R²: {r2:.3f}')

    # Residuals
    residuals = y_pred - y_test
    axes[1,0].scatter(y_pred, residuals, alpha=0.6)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Fantasy Points')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residual Plot')

    # Error distribution
    axes[1,1].hist(np.abs(residuals), bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Absolute Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution')

    plt.tight_layout()
    os.makedirs(f'../model_analysis/{year}', exist_ok=True)
    plt.savefig(f'../model_analysis/{year}/{position}_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visual(model_stats, year):
    """Create summary comparison across positions"""
    positions = list(model_stats.keys())
    maes = [model_stats[pos]['mae'] for pos in positions]
    r2s = [model_stats[pos]['r2'] for pos in positions]
    samples = [model_stats[pos]['samples'] for pos in positions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Summary', fontsize=16)

    # MAE comparison
    axes[0].bar(positions, maes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('MAE by Position')
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center')

    # R² comparison
    axes[1].bar(positions, r2s, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² by Position')
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Sample sizes
    axes[2].bar(positions, samples, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[2].set_ylabel('Training Samples')
    axes[2].set_title('Sample Size by Position')
    for i, v in enumerate(samples):
        axes[2].text(i, v + 10, f'{v}', ha='center')

    plt.tight_layout()
    os.makedirs(f'../model_analysis/{year}', exist_ok=True)
    plt.savefig(f'../model_analysis/{year}/model_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Building position-specific models for {YEAR}...")
    print()

    lagged_data = build_lagged_dataset(all_data)

    positions = ['QB', 'RB', 'WR', 'TE']
    model_stats = {}

    for position in positions:
        result = fit_position_model(lagged_data, position)
        if result is None:
            continue

        print(f"{position}: MAE = {result['mae']:.1f}, R² = {result['r2']:.3f}, "
              f"Samples = {result['samples']}, Features = {len(result['feature_cols'])}")
        print(f"Top 5 features for {position}:")
        for _, row in result['feature_importance'].head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        create_position_visuals(position, YEAR, result['feature_importance'],
                                 result['y_test'], result['y_pred'], result['mae'], result['r2'])

        model_data = {
            'model': result['model'],
            'feature_names': result['feature_cols'],
            'feature_importance': result['feature_importance'],
        }
        os.makedirs(f'../models/{YEAR}', exist_ok=True)
        with open(f'../models/{YEAR}/model_{position.lower()}.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        model_stats[position] = {'mae': result['mae'], 'r2': result['r2'], 'samples': result['samples']}

    create_summary_visual(model_stats, YEAR)

    print("\nPosition-specific models created!")
    print(f"Models saved in /models/{YEAR}/ directory as: model_qb.pkl, model_rb.pkl, etc.")
    print(f"Visualizations saved in model_analysis/{YEAR}/ directory")

if __name__ == "__main__":
    main()
