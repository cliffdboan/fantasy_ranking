import pandas as pd
from datetime import datetime

def populate_team_attempts(year):
    """Populate team context with pass/rush attempt projections"""

    # Team mapping for consistency
    team_mapping = {
        'KC': 'KAN',
        'LV': 'LVR',
        'TB': 'TAM',
        'JAC': 'JAX',
        'NE': 'NWE',
        'NO': 'NOR',
        'SF': 'SFO',
        'GB': 'GNB'
    }

    # Handle pass attempts
    pass_data = pd.read_csv(f'{year}/team_pass_attempts_{year}.csv')
    pass_data = pass_data.dropna(subset=['Player'])
    pass_data = pass_data[pass_data['Player'] != '']
    pass_data['Team_Clean'] = pass_data['Team'].str.strip().replace(team_mapping)
    pass_data['Pass Attempts'] = pd.to_numeric(pass_data['Pass Attempts'], errors='coerce')
    team_pass = pass_data.groupby('Team_Clean')['Pass Attempts'].sum().reset_index()

    # Handle rush attempts
    rush_data = pd.read_csv(f'{year}/team_rush_attempts_{year}.csv')
    rush_data = rush_data.dropna(subset=['Player'])
    rush_data = rush_data[rush_data['Player'] != '']
    rush_data['Team_Clean'] = rush_data['Team'].str.strip().replace(team_mapping)
    rush_data['Rushing Attempts'] = pd.to_numeric(rush_data['Rushing Attempts'], errors='coerce')
    team_rush = rush_data.groupby('Team_Clean')['Rushing Attempts'].sum().reset_index()

    # Load existing team context
    team_context = pd.read_csv(f'{year}/team_context_{year}.csv')

    # Merge pass attempts
    team_context = team_context.merge(
        team_pass.rename(columns={'Team_Clean': 'Team', 'Pass Attempts': 'Pass_Proj'}),
        on='Team', how='left'
    )

    # Merge rush attempts
    team_context = team_context.merge(
        team_rush.rename(columns={'Team_Clean': 'Team', 'Rushing Attempts': 'Rush_Proj'}),
        on='Team', how='left'
    )

    # Update the projection columns
    team_context['Pass_Attempts_Proj'] = team_context['Pass_Proj'].fillna(team_context['Pass_Attempts_Proj']).round(1).astype(float)
    team_context['Rush_Attempts_Proj'] = team_context['Rush_Proj'].fillna(team_context['Rush_Attempts_Proj']).round(1).astype(float)

    # Drop temporary columns
    team_context = team_context.drop(['Pass_Proj', 'Rush_Proj'], axis=1)

    # Save updated file
    team_context.to_csv(f'{year}/team_context_{year}.csv', index=False)

    print("Team context updated with attempt projections!")
    print("\nSample of updated data:")
    print(team_context[['Team', 'Pass_Attempts_Proj', 'Rush_Attempts_Proj']].head(10))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python populate_team_attempts.py <year>")

    populate_team_attempts(year)
