import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

PREDICTION_YEAR = 2025

def create_prediction_visuals(merged):
    """Create comprehensive visualizations for prediction analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Fantasy Football Prediction Analysis ({PREDICTION_YEAR} Validation)', fontsize=16)

    # Overall scatter plot
    axes[0,0].scatter(merged['PPR'], merged['Predicted_Fantasy_Points'], alpha=0.6)
    axes[0,0].plot([merged['PPR'].min(), merged['PPR'].max()],
                   [merged['PPR'].min(), merged['PPR'].max()], 'r--')
    axes[0,0].set_xlabel('Actual PPR Points')
    axes[0,0].set_ylabel('Predicted Points')
    axes[0,0].set_title('Predictions vs Actual')

    # Error distribution
    merged['Error'] = merged['Predicted_Fantasy_Points'] - merged['PPR']
    axes[0,1].hist(merged['Error'], bins=30, alpha=0.7, color='skyblue')
    axes[0,1].axvline(x=0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Prediction Error')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Error Distribution')

    # Position MAE comparison
    positions = ['QB', 'RB', 'WR', 'TE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    pos_stats = []
    for pos in positions:
        pos_data = merged[merged['FantPos'] == pos]
        if len(pos_data) > 0:
            mae = np.mean(np.abs(pos_data['Error']))
            pos_stats.append(mae)
        else:
            pos_stats.append(0)

    axes[0,2].bar(positions, pos_stats, color=colors)
    axes[0,2].set_ylabel('Mean Absolute Error')
    axes[0,2].set_title('MAE by Position')
    for i, v in enumerate(pos_stats):
        if v > 0:
            axes[0,2].text(i, v + 1, f'{v:.1f}', ha='center')

    # Top performers comparison
    top_20_actual = set(merged.nlargest(20, 'PPR')['Player'])
    top_20_pred = set(merged.nsmallest(20, 'Rank')['Player'])
    overlap = len(top_20_actual & top_20_pred)

    categories = ['Correctly\nPredicted', 'Missed']
    values = [overlap, 20 - overlap]
    axes[1,0].pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title(f'Top 20 Success Rate\n({overlap}/20 players)')

    # Residuals plot
    axes[1,1].scatter(merged['Predicted_Fantasy_Points'], merged['Error'], alpha=0.6)
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    axes[1,1].set_xlabel('Predicted Points')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residual Plot')

    # Position scatter with colors
    for i, pos in enumerate(positions):
        pos_data = merged[merged['FantPos'] == pos]
        if len(pos_data) > 0:
            axes[1,2].scatter(pos_data['PPR'], pos_data['Predicted_Fantasy_Points'],
                            label=pos, alpha=0.7, color=colors[i])
    axes[1,2].plot([merged['PPR'].min(), merged['PPR'].max()],
                   [merged['PPR'].min(), merged['PPR'].max()], 'r--')
    axes[1,2].set_xlabel('Actual PPR Points')
    axes[1,2].set_ylabel('Predicted Points')
    axes[1,2].set_title('Predictions by Position')
    axes[1,2].legend()

    plt.tight_layout()
    plt.savefig('model_analysis/ppr_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def check_ppr_model_performance():
    print("\n" + "="*60)
    print("FANTASY FOOTBALL MODEL VALIDATION")
    print(f"Predictions: {PREDICTION_YEAR} | Validation: {PREDICTION_YEAR}")
    print("="*60)
    # Load position-specific PPR predictions
    predictions = pd.read_csv(f'predictions/fantasy_predictions_position_specific_{PREDICTION_YEAR}.csv')

    # Load actual year's results - handle multi-level headers
    actual = pd.read_csv(f'stats/fantasy_stats_for_{PREDICTION_YEAR}.csv', header=1)

    # Clean actual data
    actual_clean = actual[['Player', 'FantPos', 'PPR']].copy()
    actual_clean['Player'] = actual_clean['Player'].str.replace(r'[*+]', '', regex=True)
    actual_clean['PPR'] = pd.to_numeric(actual_clean['PPR'], errors='coerce')
    actual_clean = actual_clean.dropna()

    # Merge predictions with actual PPR results
    merged = predictions.merge(actual_clean, on='Player', how='inner')
    merged = merged.dropna()

    print(f"\nDATASET OVERVIEW")
    print(f"   • Total predictions: {len(predictions):,}")
    print(f"   • Matched players: {len(merged):,}")
    print(f"   • Match rate: {len(merged)/len(predictions)*100:.1f}%")

    # Overall correlation
    if len(merged) > 1:
        pearson_corr, _ = pearsonr(merged['Predicted_Fantasy_Points'], merged['PPR'])
        spearman_corr, _ = spearmanr(merged['Predicted_Fantasy_Points'], merged['PPR'])
        mae = np.mean(np.abs(merged['Predicted_Fantasy_Points'] - merged['PPR']))
        # MAPE only for players with >50 PPR to avoid division by small numbers
        mape_data = merged[merged['PPR'] > 50]
        mape = np.mean(np.abs((mape_data['PPR'] - mape_data['Predicted_Fantasy_Points']) / mape_data['PPR'])) * 100 if len(mape_data) > 0 else 0


        print(f"\nOVERALL PERFORMANCE")
        print(f"   • Pearson Correlation: {pearson_corr:.3f} {'[GOOD]' if pearson_corr > 0.5 else '[FAIR]' if pearson_corr > 0.3 else '[POOR]'}")
        print(f"   • Spearman Correlation: {spearman_corr:.3f} {'[GOOD]' if spearman_corr > 0.5 else '[FAIR]' if spearman_corr > 0.3 else '[POOR]'}")
        print(f"   • Mean Absolute Error: {mae:.1f} points")
        print(f"   • Mean Absolute Percentage Error: {mape:.1f}%")

    # Top performers analysis
    print(f"\nTOP PERFORMERS ANALYSIS")
    top_20_predicted = merged.nsmallest(20, 'Rank')['Player'].tolist()
    top_20_actual = merged.nlargest(20, 'PPR')['Player'].tolist()
    overlap = len(set(top_20_predicted) & set(top_20_actual))
    success_rate = overlap/20*100

    print(f"   • Top 20 Success Rate: {overlap}/20 ({success_rate:.1f}%) {'[GOOD]' if success_rate > 60 else '[FAIR]' if success_rate > 40 else '[POOR]'}")

    print(f"\nTOP 5 ACTUAL PERFORMERS")
    top_actual = merged.nlargest(5, 'PPR')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Rank']]
    for i, (_, row) in enumerate(top_actual.iterrows(), 1):
        error = row['Predicted_Fantasy_Points'] - row['PPR']
        error_status = '[GOOD]' if abs(error) < 20 else '[FAIR]' if abs(error) < 40 else '[POOR]'
        print(f"   {i}. {row['Player']} ({row['FantPos']}) {error_status}")
        print(f"      Predicted: {row['Predicted_Fantasy_Points']:.0f} (Rank #{row['Rank']}) | Actual: {row['PPR']:.0f}")

    print(f"\nTOP 5 PREDICTED PERFORMERS")
    top_predicted = merged.nsmallest(5, 'Rank')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Rank']]
    for i, (_, row) in enumerate(top_predicted.iterrows(), 1):
        error = row['Predicted_Fantasy_Points'] - row['PPR']
        error_status = '[GOOD]' if abs(error) < 20 else '[FAIR]' if abs(error) < 40 else '[POOR]'
        print(f"   {i}. {row['Player']} ({row['FantPos']}) {error_status}")
        print(f"      Predicted: {row['Predicted_Fantasy_Points']:.0f} (Rank #{row['Rank']}) | Actual: {row['PPR']:.0f}")

    # Position breakdown
    print(f"\nPOSITION BREAKDOWN")
    positions = ['QB', 'RB', 'WR', 'TE']
    for pos in positions:
        pos_data = merged[merged['FantPos'] == pos]
        if len(pos_data) > 0:
            mae = np.mean(np.abs(pos_data['Predicted_Fantasy_Points'] - pos_data['PPR']))
            corr, _ = pearsonr(pos_data['Predicted_Fantasy_Points'], pos_data['PPR']) if len(pos_data) > 1 else (0, 1)
            corr_status = '[GOOD]' if corr > 0.5 else '[FAIR]' if corr > 0.3 else '[POOR]'
            mae_status = '[GOOD]' if mae < 25 else '[FAIR]' if mae < 40 else '[POOR]'
            print(f"   {pos}: {len(pos_data):2d} players | MAE: {mae:5.1f} {mae_status} | Correlation: {corr:5.3f} {corr_status}")

    # Biggest hits and misses
    merged['Error'] = merged['Predicted_Fantasy_Points'] - merged['PPR']

    print(f"\nBIGGEST HITS (Model Undervalued)")
    hits = merged.nsmallest(5, 'Error')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Error']]
    for i, (_, row) in enumerate(hits.iterrows(), 1):
        print(f"   {i}. {row['Player']} ({row['FantPos']})")
        print(f"      Predicted: {row['Predicted_Fantasy_Points']:.0f} | Actual: {row['PPR']:.0f} | Missed by: {abs(row['Error']):.0f} pts")

    print(f"\nBIGGEST MISSES (Model Overvalued)")
    misses = merged.nlargest(5, 'Error')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Error']]
    for i, (_, row) in enumerate(misses.iterrows(), 1):
        print(f"   {i}. {row['Player']} ({row['FantPos']})")
        print(f"      Predicted: {row['Predicted_Fantasy_Points']:.0f} | Actual: {row['PPR']:.0f} | Overvalued by: {row['Error']:.0f} pts")

    # Create visualizations
    create_prediction_visuals(merged)

    print(f"\nVisualizations saved to: model_analysis/ppr_prediction_analysis.png")
    print(f"\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    # Ensure model_analysis directory exists
    os.makedirs('model_analysis', exist_ok=True)
    check_ppr_model_performance()
