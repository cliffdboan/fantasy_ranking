import pandas as pd

def calculate_team_rz_rates(year=2025):
    """Calculate proper red zone pass rates using both pass and rush data"""
    
    # Load red zone pass data
    pass_data = pd.read_csv(f'{year}/redzone_pass_{year-1}.csv')
    
    # Load red zone rush data (skip header row)
    rush_data = pd.read_csv(f'{year}/redzone_rush_{year-1}.csv', skiprows=1)
    
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
        # Find the row index for this team in rush data
        team_idx = None
        for i, tm in enumerate(teams):
            if tm == team:
                team_idx = i
                break
        
        if team_idx is not None and team_idx < len(rush_data):
            total_rz_rush_att = pd.to_numeric(rush_data.iloc[team_idx, 0], errors='coerce')  # First column is Inside 20 rush attempts
            if pd.isna(total_rz_rush_att):
                total_rz_rush_att = 0
        else:
            total_rz_rush_att = 0
        
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
    team_context = team_context.merge(
        team_rz[['Team', 'RZ_Pass_Rate']].rename(columns={'RZ_Pass_Rate': 'RZ_Rate_2024'}),
        on='Team', how='left'
    )

    # Update RZ_Pass_Rate with 2024 data where available (round to 3 decimal places)
    team_context['RZ_Pass_Rate'] = team_context['RZ_Rate_2024'].fillna(team_context['RZ_Pass_Rate']).round(3)

    # Drop temporary column
    team_context = team_context.drop('RZ_Rate_2024', axis=1)

    # Save updated team context
    team_context.to_csv(f'{year}/team_context_{year}.csv', index=False)

    print("Proper red zone pass rates calculated!")
    print("\nTeam RZ Analysis:")
    print(team_rz.round(3))
    
    print(f"\nUpdated team context:")
    print(team_context[['Team', 'RZ_Pass_Rate']].head(10).round(3))

if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    calculate_team_rz_rates(year)
