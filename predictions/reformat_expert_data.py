#!/usr/bin/env python3
"""
Reformat expert projections data for easier use
"""

import pandas as pd
import numpy as np

def reformat_expert_projections(input_file, output_file):
    """Clean and reformat expert projections"""

    # Read the expert data
    df = pd.read_csv(input_file)

    # Create clean player names
    df['Player'] = df['first_name'] + ' ' + df['last_name']

    # Standardize team names
    team_mapping = {
        'KCC': 'KAN', 'TBB': 'TAM', 'GBP': 'GNB', 'JAC': 'JAX',
        'NEP': 'NWE', 'NOS': 'NOR'
    }
    df['Team'] = df['team'].map(team_mapping).fillna(df['team'])

    # Select and rename columns
    expert_clean = df[[
        'Player', 'pos', 'Team', 'age', 'points', 'rank', 'pos_rank',
        'overall_ecr', 'pos_ecr', 'adp', 'floor', 'ceiling', 'tier'
    ]].copy()

    expert_clean.columns = [
        'Player', 'Position', 'Team', 'Age', 'Expert_Points', 'Expert_Overall_Rank',
        'Expert_Pos_Rank', 'Expert_ECR', 'Expert_Pos_ECR', 'Expert_ADP',
        'Expert_Floor', 'Expert_Ceiling', 'Expert_Tier'
    ]

    # Round numerical columns
    numeric_cols = ['Expert_Points', 'Expert_Floor', 'Expert_Ceiling', 'Expert_ADP']
    for col in numeric_cols:
        expert_clean[col] = expert_clean[col].round(1)

    # Check for duplicates before removing
    initial_count = len(expert_clean)
    
    # Remove duplicates based on Player and Position
    expert_clean = expert_clean.drop_duplicates(subset=['Player', 'Position'], keep='first')
    final_count = len(expert_clean)
    
    # Sort by expert overall rank
    expert_clean = expert_clean.sort_values('Expert_Overall_Rank').reset_index(drop=True)

    # Save formatted data
    expert_clean.to_csv(output_file, index=False)
    print(f"Reformatted expert data saved to: {output_file}")

    # Print summary
    print(f"\nTotal players: {final_count}")
    if initial_count > final_count:
        print(f"Removed {initial_count - final_count} duplicate entries")
    print("\nBy position:")
    pos_summary = expert_clean['Position'].value_counts()
    for pos, count in pos_summary.items():
        print(f"  {pos}: {count}")

    return expert_clean

if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python reformat_expert_data.py <year>")

    input_file = f"expert_projections_{year}.csv"
    output_file = "expert_projections_clean.csv"

    reformat_expert_projections(input_file, output_file)
