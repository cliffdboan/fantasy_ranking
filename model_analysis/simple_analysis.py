import pandas as pd
import numpy as np

# Change date for whatever years you want to analyze
YEAR = 2023
PREDICTION_YEAR = 2024

# Load improved predictions and actual data
predictions = pd.read_csv(f'../predictions/improved_fantasy_predictions_position_specific_{PREDICTION_YEAR}.csv')
actual = pd.read_csv(f'../stats/fantasy_stats_for_{YEAR}.csv')

# Clean data - use PPR scoring
actual_clean = actual[['Player', 'FantPos', 'PPR']].copy()
actual_clean['Player'] = actual_clean['Player'].str.replace(r'[*+]', '', regex=True)
actual_clean['PPR'] = pd.to_numeric(actual_clean['PPR'], errors='coerce')

# Remove rows with NaN values
actual_clean = actual_clean.dropna()

# Merge
merged = predictions.merge(actual_clean, on='Player', how='inner')
merged = merged.dropna()

print(f"Clean dataset: {len(merged)} players")

# Calculate correlations on clean data
from scipy.stats import pearsonr, spearmanr

if len(merged) > 1:
    pearson_corr, pearson_p = pearsonr(merged['Predicted_Fantasy_Points'], merged['PPR'])
    spearman_corr, spearman_p = spearmanr(merged['Predicted_Fantasy_Points'], merged['PPR'])

    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")

# Top 10 actual vs predicted
print("\n=== TOP 10 ACTUAL PERFORMERS ===")
top_actual = merged.nlargest(10, 'PPR')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Rank']]
for _, row in top_actual.iterrows():
    print(f"{row['Player']} ({row['FantPos']}): Predicted {row['Predicted_Fantasy_Points']:.0f} (Rank {row['Rank']}), Actual {row['PPR']:.0f}")

print("\n=== TOP 10 PREDICTED PERFORMERS ===")
top_predicted = merged.nsmallest(10, 'Rank')[['Player', 'FantPos', 'Predicted_Fantasy_Points', 'PPR', 'Rank']]
for _, row in top_predicted.iterrows():
    print(f"{row['Player']} ({row['FantPos']}): Predicted {row['Predicted_Fantasy_Points']:.0f} (Rank {row['Rank']}), Actual {row['PPR']:.0f}")

# Success rate for top predictions
top_20_predicted = merged.nsmallest(20, 'Rank')['Player'].tolist()
top_20_actual = merged.nlargest(20, 'PPR')['Player'].tolist()
overlap = len(set(top_20_predicted) & set(top_20_actual))
print(f"\nTop 20 Success Rate: {overlap}/20 ({overlap/20*100:.1f}%)")

# Position breakdown
print("\n=== POSITION BREAKDOWN ===")
for pos in ['QB', 'RB', 'WR', 'TE']:
    pos_data = merged[merged['FantPos'] == pos]
    if len(pos_data) > 0:
        mae = np.mean(np.abs(pos_data['Predicted_Fantasy_Points'] - pos_data['PPR']))
        corr, _ = pearsonr(pos_data['Predicted_Fantasy_Points'], pos_data['PPR']) if len(pos_data) > 1 else (0, 1)
        print(f"{pos}: {len(pos_data)} players, MAE = {mae:.1f}, Correlation = {corr:.3f}")
