import pandas as pd

def populate_team_context(year=2025):
    """Populate team context file with defensive rankings"""

    # Load defensive rankings
    def_rankings = pd.read_csv(f'{year}/nfl_defense_rankings_{year}.csv')

    # Load existing team context template
    team_context = pd.read_csv(f'{year}/team_context_{year}.csv')

    # Create team mapping for merging
    team_mapping = {
        'GB': 'GNB',
        'KC': 'KAN',
        'LV': 'LVR',
        'NE': 'NWE',
        'NO': 'NOR',
        'SF': 'SFO',
        'TB': 'TAM'
    }

    # Apply team mapping to defensive rankings
    def_rankings['team_mapped'] = def_rankings['team'].replace(team_mapping)

    # Merge defensive rankings into team context
    updated_context = team_context.merge(
        def_rankings[['team_mapped', 'QB', 'RB', 'WR', 'TE']],
        left_on='Team',
        right_on='team_mapped',
        how='left'
    )

    # Update defensive ranking columns
    updated_context['Def_Rank_vs_QB'] = updated_context['QB']
    updated_context['Def_Rank_vs_RB'] = updated_context['RB']
    updated_context['Def_Rank_vs_WR'] = updated_context['WR']
    updated_context['Def_Rank_vs_TE'] = updated_context['TE']

    # Drop merge columns
    updated_context = updated_context.drop(['team_mapped', 'QB', 'RB', 'WR', 'TE'], axis=1)

    # Save updated file
    updated_context.to_csv(f'{year}/team_context_{year}.csv', index=False)

    print("Team context updated with defensive rankings")
    print("\nSample of updated data:")
    print(updated_context[['Team', 'Def_Rank_vs_QB', 'Def_Rank_vs_RB', 'Def_Rank_vs_WR', 'Def_Rank_vs_TE']].head())

if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    populate_team_context(year)
