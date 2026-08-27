#!/usr/bin/env python3
"""
Remove duplicate players from draft sheet CSV
Keeps the entry with the lowest Model_Rank (best ranking) for each player
"""

import pandas as pd
import re

def remove_duplicates(year):
    # Read the CSV
    df = pd.read_csv(f'draft_sheet_{year}.csv')
    
    print(f"Original CSV has {len(df)} rows")
    
    # Clean player names for comparison (remove suffixes and special characters)
    name_mappings = {
        'Cam Ward': 'Cameron Ward'
    }
    
    def clean_name(name):
        # Remove suffixes and special characters
        cleaned = str(name).replace('*', '').replace('+', '')
        cleaned = re.sub(r'\s+(Jr\.|Sr\.|III|II)$', '', cleaned)
        # Normalize initials (D.J. -> DJ, A.J. -> AJ, etc.)
        cleaned = re.sub(r'([A-Z])\.([A-Z])\.', r'\1\2', cleaned)
        # Apply name mappings
        if cleaned in name_mappings:
            cleaned = name_mappings[cleaned]
        return cleaned.strip()
    
    df['Player_Clean'] = df['Player'].apply(clean_name)
    
    # Show duplicates before removal
    duplicates = df[df.duplicated(subset=['Player_Clean'], keep=False)].sort_values(['Player_Clean', 'Model_Rank'])
    if not duplicates.empty:
        print(f"\nFound {len(duplicates)} duplicate entries for {duplicates['Player_Clean'].nunique()} players:")
        for player in duplicates['Player_Clean'].unique():
            player_dupes = duplicates[duplicates['Player_Clean'] == player]
            print(f"  {player}: {len(player_dupes)} entries (ranks: {list(player_dupes['Model_Rank'])})")
    
    # Keep only the entry with the lowest Model_Rank (best ranking) for each player
    df_clean = df.loc[df.groupby('Player_Clean')['Model_Rank'].idxmin()].copy()
    
    # Drop the helper column
    df_clean = df_clean.drop('Player_Clean', axis=1)
    
    # Sort by Model_Rank
    df_clean = df_clean.sort_values('Model_Rank').reset_index(drop=True)
    
    print(f"Cleaned CSV has {len(df_clean)} rows")
    print(f"Removed {len(df) - len(df_clean)} duplicate entries")
    
    # Save the cleaned CSV
    df_clean.to_csv(f'draft_sheet_{year}.csv', index=False)
    print("✅ Duplicates removed and CSV updated!")

if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python remove_duplicates.py <year>")

    remove_duplicates(year)