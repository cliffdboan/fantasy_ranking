import pandas as pd
import numpy as np

def add_team_context_features(data, year=2024, verbose=True):
    """Add team context features to player data"""

    # Load team context data
    try:
        team_context = pd.read_csv(f'../team_data/{year}/team_context_{year}.csv')
    except FileNotFoundError:
        if verbose:
            print(f"Team context file for {year} not found. Using defaults.")
        return data
    
    # Merge team context with player data
    data_with_context = data.merge(team_context, left_on='Tm', right_on='Team', how='left')
    
    # Create position-specific team features
    
    # QB Features with 1-QB league adjustments
    data_with_context['QB_Team_Pass_Vol'] = data_with_context['Pass_Attempts_Proj'] / 17
    data_with_context['QB_OL_Boost'] = np.where(data_with_context['OL_Rank'] <= 10, 1.1, 
                                               np.where(data_with_context['OL_Rank'] >= 25, 0.9, 1.0))
    data_with_context['QB_Coaching_Penalty'] = np.where(
        (data_with_context['HC_Change'] == 1) | (data_with_context['OC_Change'] == 1), 0.95, 1.0
    )
    
    # 1-QB league position value penalty
    data_with_context['QB_Position_Penalty'] = np.where(
        data_with_context['FantPos'] == 'QB', 0.85, 1.0  # Reduce QB values by 15%
    )
    
    # RB Features with workload premium
    data_with_context['RB_Team_Rush_Vol'] = data_with_context['Rush_Attempts_Proj'] / 17
    data_with_context['RB_OL_Boost'] = np.where(data_with_context['OL_Rank'] <= 8, 1.15,
                                               np.where(data_with_context['OL_Rank'] >= 28, 0.85, 1.0))
    
    # Workload premium for proven RBs
    data_with_context['RB_Workload_Premium'] = np.where(
        (data_with_context['FantPos'] == 'RB') & 
        (pd.to_numeric(data_with_context.get('Total_Touches', 0), errors='coerce') > 200), 1.1, 1.0
    )
    
    # WR Features
    data_with_context['WR_Team_Pass_Vol'] = data_with_context['Pass_Attempts_Proj'] / 17
    data_with_context['WR_RZ_Opportunity'] = data_with_context['RZ_Pass_Rate'] * data_with_context['Pass_Attempts_Proj'] / 17
    
    # TE Features
    data_with_context['TE_RZ_Opportunity'] = data_with_context['RZ_Pass_Rate'] * data_with_context['Pass_Attempts_Proj'] / 17
    
    # Defensive strength adjustments (lower rank = tougher defense = penalty)
    data_with_context['QB_Schedule_Difficulty'] = (33 - data_with_context['Def_Rank_vs_QB']) / 32
    data_with_context['RB_Schedule_Difficulty'] = (33 - data_with_context['Def_Rank_vs_RB']) / 32  
    data_with_context['WR_Schedule_Difficulty'] = (33 - data_with_context['Def_Rank_vs_WR']) / 32
    data_with_context['TE_Schedule_Difficulty'] = (33 - data_with_context['Def_Rank_vs_TE']) / 32
    
    return data_with_context

def get_team_context_features(position):
    """Get list of team context features for each position"""
    
    base_features = ['HC_Change', 'OC_Change']
    
    if position == 'QB':
        return base_features + [
            'QB_Team_Pass_Vol', 'QB_OL_Boost', 'QB_Coaching_Penalty', 'QB_Schedule_Difficulty', 'QB_Position_Penalty'
        ]
    elif position == 'RB':
        return base_features + [
            'RB_Team_Rush_Vol', 'RB_OL_Boost', 'RB_Schedule_Difficulty', 'RB_Workload_Premium'
        ]
    elif position == 'WR':
        return base_features + [
            'WR_Team_Pass_Vol', 'WR_RZ_Opportunity', 'WR_Schedule_Difficulty'
        ]
    elif position == 'TE':
        return base_features + [
            'TE_RZ_Opportunity', 'TE_Schedule_Difficulty'
        ]
    else:
        return base_features

if __name__ == "__main__":
    # Test the integration
    print("Team context integration ready!")
    print("Features by position:")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        features = get_team_context_features(pos)
        print(f"{pos}: {len(features)} features - {features[:3]}...")