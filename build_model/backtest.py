"""
Walk-forward backtest: for each test year Y, trains position models using
ONLY data whose target season is strictly before Y (no peeking at the
future), builds prediction features from year Y-1's actual stats - the exact
same code path predict_position_specific.py uses in production - and scores
the predictions against year Y's actual results.

This is the yardstick for whether a modeling change actually helps, rather
than a single spot-checked year.

Usage: python backtest.py [start_year] [end_year]   (defaults to 2010 2025)
"""
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.append('..')
from data_cleaning.create_df import all_data, load_fantasy_stats
from model_core import build_lagged_dataset, fit_position_model, build_prediction_features

POSITIONS = ['QB', 'RB', 'WR', 'TE']


def evaluate_year(lagged_data, test_year):
    """Train on data strictly before test_year, predict test_year, score vs actual."""
    try:
        source_data = load_fantasy_stats(f'../stats/fantasy_stats_for_{test_year - 1}.csv')
    except FileNotFoundError:
        return []
    try:
        prev_source_data = load_fantasy_stats(f'../stats/fantasy_stats_for_{test_year - 2}.csv')
    except FileNotFoundError:
        prev_source_data = None
    try:
        actual_data = load_fantasy_stats(f'../stats/fantasy_stats_for_{test_year}.csv')
    except FileNotFoundError:
        return []

    actual_clean = actual_data[['Player', 'FantPos', 'PPR']].copy()
    actual_clean['Player'] = actual_clean['Player'].str.replace(r'[*+]', '', regex=True)
    actual_clean['PPR'] = pd.to_numeric(actual_clean['PPR'], errors='coerce')
    actual_clean = actual_clean.dropna(subset=['PPR'])

    rows = []
    for position in POSITIONS:
        fit = fit_position_model(lagged_data, position, as_of_year=test_year)
        if fit is None:
            continue

        pos_data, features_df = build_prediction_features(
            source_data, test_year, position, fit['feature_cols'], prev_source_data=prev_source_data
        )
        if features_df is None or len(pos_data) == 0:
            continue

        preds = fit['model'].predict(features_df.values)
        pred_df = pd.DataFrame({
            'Player': pos_data['Player'].str.replace(r'[*+]', '', regex=True).values,
            'Predicted': preds,
        })

        merged = pred_df.merge(actual_clean, on='Player', how='inner')
        merged = merged.dropna(subset=['PPR'])
        if len(merged) < 5:
            continue

        merged['Rank'] = merged['Predicted'].rank(ascending=False)
        mae = float(np.mean(np.abs(merged['Predicted'] - merged['PPR'])))
        pearson_corr = float(pearsonr(merged['Predicted'], merged['PPR'])[0]) if len(merged) > 1 else np.nan
        spearman_corr = float(spearmanr(merged['Predicted'], merged['PPR'])[0]) if len(merged) > 1 else np.nan

        if len(merged) >= 20:
            top_actual = set(merged.nlargest(20, 'PPR')['Player'])
            top_pred = set(merged.nsmallest(20, 'Rank')['Player'])
            hit_rate = len(top_actual & top_pred) / 20
        else:
            hit_rate = np.nan

        rows.append({
            'Year': test_year,
            'Position': position,
            'Players': len(merged),
            'MAE': mae,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'Top20_Hit_Rate': hit_rate,
        })

    return rows


def run_backtest(start_year, end_year):
    print("Loading historical data...")
    lagged_data = build_lagged_dataset(all_data)
    print(f"{len(lagged_data)} lagged (season -> next-season) training pairs available\n")

    all_rows = []
    for year in range(start_year, end_year + 1):
        print(f"Backtesting {year}...")
        rows = evaluate_year(lagged_data, year)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def summarize(results):
    if results.empty:
        print("No results.")
        return

    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST SUMMARY")
    print("=" * 70)

    by_position = results.groupby('Position').agg(
        Years=('Year', 'nunique'),
        Avg_MAE=('MAE', 'mean'),
        Avg_Pearson=('Pearson', 'mean'),
        Avg_Spearman=('Spearman', 'mean'),
        Avg_Top20_Hit_Rate=('Top20_Hit_Rate', 'mean'),
    ).round(3)
    print("\nBy position (averaged across all tested years):")
    print(by_position)

    print(f"\nOverall across {results['Year'].nunique()} years, {len(results)} position-year folds:")
    print(f"  Avg MAE:            {results['MAE'].mean():.1f}")
    print(f"  Avg Pearson corr:   {results['Pearson'].mean():.3f}")
    print(f"  Avg Spearman corr:  {results['Spearman'].mean():.3f}")
    print(f"  Avg Top20 hit rate: {results['Top20_Hit_Rate'].mean():.3f}")


if __name__ == "__main__":
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    results = run_backtest(start_year, end_year)

    os.makedirs('../model_analysis/backtest', exist_ok=True)
    results.to_csv('../model_analysis/backtest/backtest_results.csv', index=False)
    print("\nDetailed per-year results saved to model_analysis/backtest/backtest_results.csv")

    summarize(results)
