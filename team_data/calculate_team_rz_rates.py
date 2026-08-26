import pandas as pd
from datetime import datetime

def calculate_team_rz_rates(year):
    """Calculate proper red zone pass rates using both pass and rush data"""

    # Load red zone pass data
    pass_data = pd.read_csv(f'{year}/redzone_pass_{year-1}.csv')

    # Load red zone rush data
    rush_data = pd.read_csv(f'{year}/redzone_rush_{year-1}.csv')

    # Get team names from pass data
    teams = pass_data['Tm'].unique()

    team_rz_rates = []

    for team in teams:
        if team == '2TM':  # Skip multi-team players
            continue

        # Get team's QB red zone pass attempts (Inside 20 column)
        team_pass = pass_data[pass_data['Tm'] == team]
        total_rz_pass_att = team_pass['Att'].sum()  # Inside 20 pass attempts

        # Get team's RB red zone rush attempts (Inside 20 column)
        team_rush = rush_data[rush_data['Tm'] == team]
        total_rz_rush_att = team_rush['Att'].sum()  # Inside 20 rush attempts

        # Calculate red zone pass rate
        total_rz_plays = total_rz_pass_att + total_rz_rush_att
        if total_rz_plays > 0:
            rz_pass_rate = total_rz_pass_att / total_rz_plays
        else:
            rz_pass_rate = 0.65  # Default

        team_rz_rates.append({
            'Team': team,
            'RZ_Pass_Att': total_rz_pass_att,
            'RZ_Rush_Att': total_rz_rush_att,
            'Total_RZ_Plays': total_rz_plays,
            'RZ_Pass_Rate': rz_pass_rate
        })

    # Create DataFrame
    team_rz = pd.DataFrame(team_rz_rates)

    # Load team context
    team_context = pd.read_csv(f'{year}/team_context_{year}.csv')

    # Merge with team context
    prior_year_col = f'RZ_Rate_{year - 1}'
    team_context = team_context.merge(
        team_rz[['Team', 'RZ_Pass_Rate']].rename(columns={'RZ_Pass_Rate': prior_year_col}),
        on='Team', how='left'
    )

    # Update RZ_Pass_Rate with prior year data where available (round to 3 decimal places)
    team_context['RZ_Pass_Rate'] = team_context[prior_year_col].fillna(team_context['RZ_Pass_Rate']).round(3)

    # Drop temporary column
    team_context = team_context.drop(prior_year_col, axis=1)

    # Save updated team context
    team_context.to_csv(f'{year}/team_context_{year}.csv', index=False)

    print("Proper red zone pass rates calculated!")
    print("\nTeam RZ Analysis:")
    print(team_rz.round(3))

    print(f"\nUpdated team context:")
    print(team_context[['Team', 'RZ_Pass_Rate']].head(10).round(3))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python calculate_team_rz_rates.py <year>")

    calculate_team_rz_rates(year)
