import pandas as pd
import numpy as np

# Real team_data/{year}/team_context_{year}.csv only exists for two seasons
# (2024, 2025) out of the ~25 seasons used in training. For every other
# target year - and for any individual row whose team code doesn't match a
# row in a file that DOES exist (e.g. a multi-team code like "3TM") - there's
# no real signal to merge. Leaving those columns absent lets them fall
# through to the generic fillna(0) used downstream when the feature matrix
# is built, which is fine for flag features (0 = "no coaching change" is a
# real, meaningful value) but wrong for the multiplier/rank/volume features
# below: a QB_OL_Boost of 0 reads as "erase this player's signal entirely",
# not "unknown", and a rank of 0 isn't a real rank either. These constants
# are neutral, "average team" stand-ins instead, derived from the mean/
# median observed across the two real seasons on file:
#   OL_Rank / Def_Rank_vs_*: mean ~16.4, median 16.5 -> middle of 32 teams
#   Pass_Attempts_Proj:      mean 578.9, median 578.2
#   Rush_Attempts_Proj:      mean 370.0, median 368.3
#   RZ_Pass_Rate:            mean 0.63,  median 0.64
LEAGUE_AVG_RANK = 16.5
LEAGUE_AVG_PASS_ATTEMPTS = 578
LEAGUE_AVG_RUSH_ATTEMPTS = 370
LEAGUE_AVG_RZ_PASS_RATE = 0.63

# HC_Change / OC_Change are intentionally left out of this table: 0 ("no
# coaching change") is already a legitimate default, so the generic
# fillna(0) downstream (and the explicit fillna(0) below) already handle
# them correctly.
TEAM_CONTEXT_DEFAULTS = {
    'OL_Rank': LEAGUE_AVG_RANK,
    'Def_Rank_vs_QB': LEAGUE_AVG_RANK,
    'Def_Rank_vs_RB': LEAGUE_AVG_RANK,
    'Def_Rank_vs_WR': LEAGUE_AVG_RANK,
    'Def_Rank_vs_TE': LEAGUE_AVG_RANK,
    'Pass_Attempts_Proj': LEAGUE_AVG_PASS_ATTEMPTS,
    'Rush_Attempts_Proj': LEAGUE_AVG_RUSH_ATTEMPTS,
    'RZ_Pass_Rate': LEAGUE_AVG_RZ_PASS_RATE,
}


def add_team_context_features(data, year=2024, verbose=True):
    """Add team context features to player data.

    When a real team_data file exists for `year`, real team context is
    merged in as before. When it doesn't (~96% of training rows, since real
    team_data only covers 2024-2025), or when a specific row's team code
    doesn't match any row in a file that does exist, the merged columns are
    filled with neutral, league-average defaults (TEAM_CONTEXT_DEFAULTS)
    instead of being left missing - so every derived feature below (boosts,
    penalties, schedule difficulty, volume) resolves to its own neutral
    value (~1.0 for a multiplier, "average" for a rank/volume/rate) rather
    than the wildly out-of-distribution 0 that downstream code would
    otherwise silently fill in.

    A Has_Team_Context flag (1 = real merged data, 0 = imputed default) is
    also added so the model can itself learn how much to trust this block
    per row instead of treating imputed and observed rows identically.
    """
    try:
        team_context = pd.read_csv(f'../team_data/{year}/team_context_{year}.csv')
        data_with_context = data.merge(team_context, left_on='Tm', right_on='Team', how='left')
    except FileNotFoundError:
        if verbose:
            print(f"Team context file for {year} not found. Using league-average defaults.")
        data_with_context = data.copy()
        for col in TEAM_CONTEXT_DEFAULTS:
            data_with_context[col] = np.nan
        for col in ('HC_Change', 'OC_Change'):
            data_with_context[col] = np.nan

    # Real signal vs. imputed default, captured before any defaults are
    # filled in below (OL_Rank is only non-null after a successful merge).
    data_with_context['Has_Team_Context'] = data_with_context['OL_Rank'].notna().astype(int)

    for col, default in TEAM_CONTEXT_DEFAULTS.items():
        data_with_context[col] = data_with_context[col].fillna(default)
    for col in ('HC_Change', 'OC_Change'):
        data_with_context[col] = data_with_context[col].fillna(0)

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

    base_features = ['HC_Change', 'OC_Change', 'Has_Team_Context']
    
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