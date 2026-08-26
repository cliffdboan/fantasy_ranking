#!/usr/bin/env python3
"""
Compare model predictions with expert consensus
"""

import pandas as pd
import numpy as np

def categorize_edge(rank_diff):
    """Categorize ranking edge"""
    if pd.isna(rank_diff):
        return 'No Expert Data'
    elif rank_diff >= 50:
        return 'Strong Buy'
    elif rank_diff >= 20:
        return 'Buy'
    elif rank_diff >= 10:
        return 'Slight Buy'
    elif rank_diff <= -50:
        return 'Strong Fade'
    elif rank_diff <= -20:
        return 'Fade'
    elif rank_diff <= -10:
        return 'Slight Fade'
    else:
        return 'Consensus'

def compare_predictions(model_file, expert_file, output_file):
    """Create comparison between model and expert predictions"""
    
    # Load both datasets
    model_df = pd.read_csv(model_file)
    expert_df = pd.read_csv(expert_file)
    
    # Clean player names for better matching
    model_df['Player_Clean'] = model_df['Player'].str.replace(r'[*+]', '', regex=True)
    expert_df['Player_Clean'] = expert_df['Player'].str.replace(r'[*+]', '', regex=True)
    
    # Merge on cleaned player name and position
    comparison = model_df.merge(
        expert_df, 
        on=['Player_Clean', 'Position'], 
        how='left',  # Keep all model predictions
        suffixes=('_Model', '_Expert')
    )
    
    # Calculate differences (only where expert data exists)
    comparison['Points_Diff'] = comparison['Adjusted_Fantasy_Points'] - comparison['Expert_Points']
    comparison['Rank_Diff'] = comparison['Expert_Overall_Rank'] - comparison['Adjusted_Rank']
    comparison['Edge_Category'] = comparison['Rank_Diff'].apply(categorize_edge)
    
    # Add position-specific rank differences (if available)
    if 'Adjusted_Pos_Rank' in comparison.columns:
        comparison['Pos_Rank_Diff'] = comparison['Expert_Pos_Rank'] - comparison['Adjusted_Pos_Rank']
    else:
        comparison['Pos_Rank_Diff'] = None
    
    # Select key columns for draft sheet
    base_columns = [
        'Player_Model', 'Position', 'Team_Model', 'Age_Model',
        'Adjusted_Rank', 'Expert_Overall_Rank', 'Rank_Diff',
        'Adjusted_Fantasy_Points', 'Expert_Points', 'Points_Diff',
        'Edge_Category', 'Expert_ADP', 'Expert_Tier'
    ]
    
    # Add position rank columns if available
    if 'Expert_Pos_Rank' in comparison.columns:
        base_columns.extend(['Expert_Pos_Rank', 'Pos_Rank_Diff'])
    
    draft_sheet = comparison[base_columns].copy()
    
    # Set column names
    base_col_names = [
        'Player', 'Position', 'Team', 'Age',
        'Model_Rank', 'Expert_Rank', 'Rank_Difference',
        'Model_Points', 'Expert_Points', 'Points_Difference',
        'Edge_Category', 'ADP', 'Expert_Tier'
    ]
    
    if 'Expert_Pos_Rank' in comparison.columns:
        base_col_names.extend(['Expert_Pos_Rank', 'Pos_Rank_Difference'])
    
    draft_sheet.columns = base_col_names
    
    # Round numerical columns
    numeric_cols = ['Model_Points', 'Expert_Points', 'Points_Difference', 'ADP']
    for col in numeric_cols:
        if col in draft_sheet.columns:
            draft_sheet[col] = draft_sheet[col].round(1)
    
    # Sort by model rank
    draft_sheet = draft_sheet.sort_values('Model_Rank').reset_index(drop=True)
    
    # Save comparison
    draft_sheet.to_csv(output_file, index=False)
    print(f"Draft comparison sheet saved to: {output_file}")
    
    return draft_sheet

def print_summary(comparison_df):
    """Print comparison summary"""
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Total players: {len(comparison_df)}")
    
    # Players with expert data
    with_expert = comparison_df.dropna(subset=['Expert_Rank'])
    print(f"Players with expert data: {len(with_expert)}")
    
    if len(with_expert) > 0:
        edge_summary = with_expert['Edge_Category'].value_counts()
        print("\nEdge Categories:")
        for category, count in edge_summary.items():
            print(f"  {category}: {count}")
        
        # Position breakdown
        print("\nBy Position:")
        pos_summary = with_expert.groupby('Position')['Edge_Category'].value_counts()
        for (pos, category), count in pos_summary.items():
            if category in ['Strong Buy', 'Buy']:
                print(f"  {pos} {category}: {count}")
    
    # Show top edge plays
    print("\n=== TOP EDGE OPPORTUNITIES ===")
    
    strong_buys = comparison_df[comparison_df['Edge_Category'] == 'Strong Buy'].head(10)
    if len(strong_buys) > 0:
        print("\nSTRONG BUYS (Model much higher than experts):")
        for _, player in strong_buys.iterrows():
            adp = f", ADP: {player['ADP']:.1f}" if pd.notna(player['ADP']) else ""
            print(f"  {player['Player']} ({player['Position']}) - Model: {int(player['Model_Rank'])}, Expert: {int(player['Expert_Rank'])}{adp}")
    
    strong_fades = comparison_df[comparison_df['Edge_Category'] == 'Strong Fade'].head(5)
    if len(strong_fades) > 0:
        print("\nSTRONG FADES (Model much lower than experts):")
        for _, player in strong_fades.iterrows():
            adp = f", ADP: {player['ADP']:.1f}" if pd.notna(player['ADP']) else ""
            print(f"  {player['Player']} ({player['Position']}) - Model: {int(player['Model_Rank'])}, Expert: {int(player['Expert_Rank'])}{adp}")

if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python compare_predictions.py <year>")

    model_file = f"../predictions/{year}/fantasy_predictions_adjusted_{year}.csv"
    expert_file = "../predictions/expert_projections_clean.csv"
    output_file = "model_vs_expert_comparison.csv"

    comparison_df = compare_predictions(model_file, expert_file, output_file)
    print_summary(comparison_df)