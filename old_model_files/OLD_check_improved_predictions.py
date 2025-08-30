import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

CHECK_YEAR = 2023
CURRENT_YEAR = CHECK_YEAR + 1

def compare_model_performance():
    # Load original predictions
    try:
        original_predictions = pd.read_csv(f'predictions/fantasy_predictions_position_specific_{CURRENT_YEAR}.csv')
        print("Original predictions loaded")
    except:
        print("Original predictions not found")
        return

    # Load improved predictions
    try:
        improved_predictions = pd.read_csv(f'predictions/improved_fantasy_predictions_position_specific_{CURRENT_YEAR}.csv')
        print("Improved predictions loaded")
    except:
        print("Improved predictions not found")
        return

    # Load actual year's results
    actual = pd.read_csv(f'stats/fantasy_stats_for_{CHECK_YEAR}.csv')

    # Clean actual data
    actual_clean = actual[['Player', 'FantPos', 'PPR']].copy()
    actual_clean['Player'] = actual_clean['Player'].str.replace(r'[*+]', '', regex=True)
    actual_clean['PPR'] = pd.to_numeric(actual_clean['PPR'], errors='coerce')
    actual_clean = actual_clean.dropna()

    # Merge with actual results
    original_merged = original_predictions.merge(actual_clean, on='Player', how='inner')
    improved_merged = improved_predictions.merge(actual_clean, on='Player', how='inner')

    print(f"\\n=== MODEL COMPARISON ===")
    print(f"Original model matched players: {len(original_merged)}")
    print(f"Improved model matched players: {len(improved_merged)}")

    # Overall performance comparison
    if len(original_merged) > 1 and len(improved_merged) > 1:
        # Original model stats
        orig_pearson, _ = pearsonr(original_merged['Predicted_Fantasy_Points'], original_merged['PPR'])
        orig_spearman, _ = spearmanr(original_merged['Predicted_Fantasy_Points'], original_merged['PPR'])
        orig_mae = np.mean(np.abs(original_merged['Predicted_Fantasy_Points'] - original_merged['PPR']))

        # Improved model stats
        imp_pearson, _ = pearsonr(improved_merged['Predicted_Fantasy_Points'], improved_merged['PPR'])
        imp_spearman, _ = spearmanr(improved_merged['Predicted_Fantasy_Points'], improved_merged['PPR'])
        imp_mae = np.mean(np.abs(improved_merged['Predicted_Fantasy_Points'] - improved_merged['PPR']))

        print(f"\\n=== OVERALL PERFORMANCE ===")
        print(f"                    Original    Improved    Change")
        print(f"Pearson Correlation:  {orig_pearson:.3f}      {imp_pearson:.3f}     {imp_pearson-orig_pearson:+.3f}")
        print(f"Spearman Correlation: {orig_spearman:.3f}      {imp_spearman:.3f}     {imp_spearman-orig_spearman:+.3f}")
        print(f"MAE:                  {orig_mae:.1f}        {imp_mae:.1f}       {imp_mae-orig_mae:+.1f}")

    # Position breakdown
    print(f"\\n=== POSITION BREAKDOWN ===")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        orig_pos = original_merged[original_merged['FantPos'] == pos]
        imp_pos = improved_merged[improved_merged['FantPos'] == pos]

        if len(orig_pos) > 1 and len(imp_pos) > 1:
            orig_mae = np.mean(np.abs(orig_pos['Predicted_Fantasy_Points'] - orig_pos['PPR']))
            imp_mae = np.mean(np.abs(imp_pos['Predicted_Fantasy_Points'] - imp_pos['PPR']))
            orig_corr, _ = pearsonr(orig_pos['Predicted_Fantasy_Points'], orig_pos['PPR'])
            imp_corr, _ = pearsonr(imp_pos['Predicted_Fantasy_Points'], imp_pos['PPR'])

            print(f"{pos}: Original MAE={orig_mae:.1f}, Improved MAE={imp_mae:.1f} (Δ{imp_mae-orig_mae:+.1f})")
            print(f"     Original Corr={orig_corr:.3f}, Improved Corr={imp_corr:.3f} (Δ{imp_corr-orig_corr:+.3f})")

    # Check specific problem cases
    print(f"\\n=== SPECIFIC IMPROVEMENTS ===")

    # Josh Allen check
    josh_allen_orig = original_merged[original_merged['Player'].str.contains('Josh Allen', na=False)]
    josh_allen_imp = improved_merged[improved_merged['Player'].str.contains('Josh Allen', na=False)]

    if len(josh_allen_orig) > 0 and len(josh_allen_imp) > 0:
        orig_pred = josh_allen_orig['Predicted_Fantasy_Points'].iloc[0]
        imp_pred = josh_allen_imp['Predicted_Fantasy_Points'].iloc[0]
        actual_points = josh_allen_orig['PPR'].iloc[0]

        print(f"Josh Allen:")
        print(f"  Actual: {actual_points:.0f}")
        print(f"  Original prediction: {orig_pred:.0f} (error: {orig_pred-actual_points:+.0f})")
        print(f"  Improved prediction: {imp_pred:.0f} (error: {imp_pred-actual_points:+.0f})")
        print(f"  Improvement: {abs(imp_pred-actual_points) - abs(orig_pred-actual_points):+.0f} points")

    # Top 20 success rate comparison
    orig_top20 = set(original_merged.nsmallest(20, 'Rank')['Player'].tolist())
    imp_top20 = set(improved_merged.nsmallest(20, 'Rank')['Player'].tolist())
    actual_top20 = set(actual_clean.nlargest(20, 'PPR')['Player'].tolist())

    orig_success = len(orig_top20 & actual_top20)
    imp_success = len(imp_top20 & actual_top20)

    print(f"\\nTop 20 Success Rate:")
    print(f"  Original: {orig_success}/20 ({orig_success/20*100:.1f}%)")
    print(f"  Improved: {imp_success}/20 ({imp_success/20*100:.1f}%)")
    print(f"  Change: {imp_success-orig_success:+d} players")

    # Show biggest improvements and regressions
    common_players = set(original_merged['Player']) & set(improved_merged['Player'])

    improvements = []
    for player in common_players:
        orig_row = original_merged[original_merged['Player'] == player].iloc[0]
        imp_row = improved_merged[improved_merged['Player'] == player].iloc[0]
        actual_points = orig_row['PPR']

        orig_error = abs(orig_row['Predicted_Fantasy_Points'] - actual_points)
        imp_error = abs(imp_row['Predicted_Fantasy_Points'] - actual_points)
        improvement = orig_error - imp_error

        improvements.append({
            'Player': player,
            'Position': orig_row['FantPos'],
            'Actual': actual_points,
            'Original_Pred': orig_row['Predicted_Fantasy_Points'],
            'Improved_Pred': imp_row['Predicted_Fantasy_Points'],
            'Improvement': improvement
        })

    improvements_df = pd.DataFrame(improvements)

    print(f"\\n=== BIGGEST IMPROVEMENTS ===")
    top_improvements = improvements_df.nlargest(5, 'Improvement')
    for _, row in top_improvements.iterrows():
        print(f"{row['Player']} ({row['Position']}): {row['Improvement']:+.0f} point error reduction")
        print(f"  Actual: {row['Actual']:.0f}, Original: {row['Original_Pred']:.0f}, Improved: {row['Improved_Pred']:.0f}")

    print(f"\\n=== BIGGEST REGRESSIONS ===")
    top_regressions = improvements_df.nsmallest(5, 'Improvement')
    for _, row in top_regressions.iterrows():
        print(f"{row['Player']} ({row['Position']}): {row['Improvement']:+.0f} point error increase")
        print(f"  Actual: {row['Actual']:.0f}, Original: {row['Original_Pred']:.0f}, Improved: {row['Improved_Pred']:.0f}")

if __name__ == "__main__":
    compare_model_performance()
