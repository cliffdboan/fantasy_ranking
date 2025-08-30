import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def check_adjusted_predictions():
    """Check performance of adjusted predictions"""

    # Load adjusted predictions
    predictions = pd.read_csv('../predictions/fantasy_predictions_adjusted_2024.csv')

    # Load actual results
    actual = pd.read_csv('../stats/fantasy_stats_for_2024.csv', header=1)
    actual_clean = actual[['Player', 'FantPos', 'PPR']].copy()
    actual_clean['Player'] = actual_clean['Player'].str.replace(r'[*+]', '', regex=True)
    actual_clean['PPR'] = pd.to_numeric(actual_clean['PPR'], errors='coerce')
    actual_clean = actual_clean.dropna()

    # Clean prediction player names
    predictions['Player_Clean'] = predictions['Player'].str.replace(r'[*+]', '', regex=True)

    # Merge
    merged = predictions.merge(actual_clean, left_on='Player_Clean', right_on='Player', how='inner')

    print("ADJUSTED MODEL PERFORMANCE")
    print("="*50)

    # Compare original vs adjusted
    orig_mae = np.mean(np.abs(merged['Predicted_Fantasy_Points'] - merged['PPR']))
    adj_mae = np.mean(np.abs(merged['Adjusted_Fantasy_Points'] - merged['PPR']))

    orig_corr, _ = pearsonr(merged['Predicted_Fantasy_Points'], merged['PPR'])
    adj_corr, _ = pearsonr(merged['Adjusted_Fantasy_Points'], merged['PPR'])

    print(f"Original MAE: {orig_mae:.1f}")
    print(f"Adjusted MAE: {adj_mae:.1f} ({'↓' if adj_mae < orig_mae else '↑'}{abs(adj_mae - orig_mae):.1f})")
    print(f"Original Correlation: {orig_corr:.3f}")
    print(f"Adjusted Correlation: {adj_corr:.3f} ({'↑' if adj_corr > orig_corr else '↓'}{abs(adj_corr - orig_corr):.3f})")

    # Top 20 success rate
    top_20_orig = set(merged.nsmallest(20, 'Rank')['Player_Clean'])
    top_20_adj = set(merged.nsmallest(20, 'Adjusted_Rank')['Player_Clean'])
    top_20_actual = set(merged.nlargest(20, 'PPR')['Player_Clean'])

    orig_success = len(top_20_orig & top_20_actual)
    adj_success = len(top_20_adj & top_20_actual)

    print(f"Original Top 20 Success: {orig_success}/20 ({orig_success/20*100:.1f}%)")
    print(f"Adjusted Top 20 Success: {adj_success}/20 ({adj_success/20*100:.1f}%)")

    # Show biggest improvements
    merged['Orig_Error'] = abs(merged['Predicted_Fantasy_Points'] - merged['PPR'])
    merged['Adj_Error'] = abs(merged['Adjusted_Fantasy_Points'] - merged['PPR'])
    merged['Error_Improvement'] = merged['Orig_Error'] - merged['Adj_Error']

    print(f"\nBIGGEST IMPROVEMENTS:")
    improvements = merged.nlargest(5, 'Error_Improvement')
    for i, (_, row) in enumerate(improvements.iterrows(), 1):
        print(f"   {i}. {row['Player_Clean']} ({row['Position']})")
        print(f"      Error: {row['Orig_Error']:.0f} → {row['Adj_Error']:.0f} (↓{row['Error_Improvement']:.0f})")
        print(f"      Actual: {row['PPR']:.0f} | Reason: {row['Adjustment_Reason']}")

if __name__ == "__main__":
    check_adjusted_predictions()
