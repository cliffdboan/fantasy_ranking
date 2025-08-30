import pandas as pd
import numpy as np

def add_contextual_features(data):
    """Add contextual features that explain major prediction errors"""

    # 1. TEAM CHANGE PENALTY
    data_sorted = data.sort_values(['Player', 'Age'])
    data_sorted['Team_Change'] = data_sorted.groupby('Player')['Tm'].transform(lambda x: (x != x.shift(1)).astype(int))
    data_sorted['Team_Change'] = data_sorted['Team_Change'].fillna(0)

    # 2. INJURY RECOVERY INDICATOR
    prev_games = pd.to_numeric(data_sorted.groupby('Player')['G'].shift(1), errors='coerce')
    data_sorted['Games_Missed_Prev'] = 17 - prev_games
    data_sorted['Games_Missed_Prev'] = data_sorted['Games_Missed_Prev'].fillna(0)
    data_sorted['Injury_Recovery'] = np.where(data_sorted['Games_Missed_Prev'] > 8, 1, 0)

    # 3. WORKLOAD PREMIUM (Fix RB undervaluation)
    # Calculate total touches for RBs
    rush_att = pd.to_numeric(data_sorted['Att.1'], errors='coerce').fillna(0)
    receptions = pd.to_numeric(data_sorted['Rec'], errors='coerce').fillna(0)
    data_sorted['Total_Touches'] = rush_att + receptions
    data_sorted['Workload_Premium'] = np.where(
        (data_sorted['FantPos'] == 'RB') & (data_sorted['Total_Touches'] > 200), 1, 0
    )
    data_sorted['Elite_Workload'] = np.where(
        (data_sorted['FantPos'] == 'RB') & (data_sorted['Total_Touches'] > 250), 1, 0
    )
    
    # Proven starter bonus for established RBs
    data_sorted['Proven_Starter'] = np.where(
        (data_sorted['FantPos'] == 'RB') & (data_sorted['Total_Touches'] > 180) & 
        (pd.to_numeric(data_sorted['G'], errors='coerce') > 12), 1, 0
    )

    # 4. POSITION VALUE ADJUSTMENTS (Fix QB overvaluation in 1-QB)
    data_sorted['QB_Streaming_Penalty'] = np.where(
        (data_sorted['FantPos'] == 'QB') & (data_sorted.groupby('FantPos').cumcount() >= 12), 1, 0
    )
    
    # 5. UNCERTAINTY DISCOUNT
    # Penalize players in unclear situations
    prev_ppr = pd.to_numeric(data_sorted.groupby('Player')['PPR'].shift(1), errors='coerce')
    data_sorted['Low_Usage_High_Potential'] = np.where(
        (prev_ppr < 100) & (data_sorted['Age'] < 26), 1, 0
    )
    data_sorted['Committee_Risk'] = np.where(
        (data_sorted['FantPos'] == 'RB') & (data_sorted['Low_Usage_High_Potential'] == 1), 1, 0
    )

    # 6. REDUCED REGRESSION PENALTIES
    # Less harsh on proven performers
    prev_ppr_high = pd.to_numeric(data_sorted.groupby('Player')['PPR'].shift(1), errors='coerce')
    prev_targets = pd.to_numeric(data_sorted.groupby('Player')['Tgt'].shift(1), errors='coerce')
    
    data_sorted['Proven_Performer'] = np.where(prev_ppr_high > 200, 1, 0)
    
    # Significantly reduce target regression risk for elite target share players
    data_sorted['Elite_Target_Share'] = np.where(
        (data_sorted['FantPos'].isin(['WR', 'TE'])) & (prev_targets > 140), 1, 0
    )
    data_sorted['Target_Regression_Risk'] = np.where(
        data_sorted['Elite_Target_Share'] == 1, 0.2,  # Minimal penalty for elite target share
        np.where((data_sorted['FantPos'].isin(['WR', 'TE'])) & (prev_ppr_high > 250), 0.4, 0)  # Reduced penalty
    )

    # 7. CONSISTENCY SCORE
    ppr_numeric = pd.to_numeric(data_sorted['PPR'], errors='coerce')
    player_stats = ppr_numeric.groupby(data_sorted['Player']).agg(['std', 'count']).reset_index()
    player_stats.columns = ['Player', 'PPR_Std', 'Seasons_Played']
    player_stats['PPR_Std'] = player_stats['PPR_Std'].fillna(0)
    data_sorted = data_sorted.merge(player_stats, on='Player', how='left')

    return data_sorted

def add_advanced_position_features(data, position):
    """Add position-specific advanced features"""

    if position == 'QB':
        # QB-specific: Capture breakout potential (Sam Darnold)
        data['Pass_Att_PG'] = pd.to_numeric(data['Att'], errors='coerce') / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)
        data['QB_Efficiency'] = pd.to_numeric(data['TD'], errors='coerce') / pd.to_numeric(data['Att'], errors='coerce').replace(0, 1) * 100
        data['Turnover_Rate'] = (pd.to_numeric(data['Int'], errors='coerce') + pd.to_numeric(data['FL'], errors='coerce')) / pd.to_numeric(data['Att'], errors='coerce').replace(0, 1) * 100

        # New starter bonus (often undervalued)
        prev_att = pd.to_numeric(data.groupby('Player')['Att'].shift(1), errors='coerce')
        data['Low_Attempts_Prev'] = np.where(prev_att < 200, 1, 0)

    elif position == 'RB':
        # RB-specific: Emphasize proven workload
        data['Snap_Share_Proxy'] = pd.to_numeric(data['G'], errors='coerce') / 17
        rush_tds = pd.to_numeric(data['TD.1'], errors='coerce').fillna(0)
        data['Goal_Line_Upside'] = rush_tds / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)
        
        # Workload sustainability bonus
        data['Workload_Bonus'] = np.where(data['Workload_Premium'] == 1, 15, 0)
        data['Elite_Workload_Bonus'] = np.where(data['Elite_Workload'] == 1, 25, 0)
        data['Proven_Starter_Bonus'] = np.where(data['Proven_Starter'] == 1, 20, 0)
        
        # Reduce uncertainty penalties for young RBs
        data['Young_Opportunity'] = np.where(
            (pd.to_numeric(data['Age'], errors='coerce') < 26) & (data['Committee_Risk'] == 0), 1, 0
        )

    elif position in ['WR', 'TE']:
        # WR/TE: Reduced regression penalties
        rec_yds = pd.to_numeric(data['Yds.2'], errors='coerce').fillna(0)
        data['Target_Efficiency'] = rec_yds / pd.to_numeric(data['Tgt'], errors='coerce').replace(0, 1)
        rec_tds = pd.to_numeric(data['TD.2'], errors='coerce').fillna(0)
        data['Red_Zone_Value'] = rec_tds / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)

        # Elite target share protection (reduce CeeDee Lamb, Amon-Ra fades)
        data['Target_Regression_Protection'] = np.where(data['Elite_Target_Share'] == 1, 25, 0)
        data['Proven_Target_Bonus'] = np.where(
            (pd.to_numeric(data.groupby('Player')['Tgt'].shift(1), errors='coerce') > 120) & 
            (data['Elite_Target_Share'] == 0), 10, 0
        )

    return data

def build_enhanced_model(position):
    """Build model with enhanced contextual features"""
    import sys
    sys.path.append('..')
    from data_cleaning.create_df import all_data

    # Load and enhance data
    data = all_data.copy()
    data['PPR_Points'] = pd.to_numeric(data['PPR'], errors='coerce')
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(25)
    data['G'] = pd.to_numeric(data['G'], errors='coerce').fillna(0)

    # Convert key numeric columns
    numeric_cols = ['Att', 'TD', 'Int', 'Tgt', 'Rec', 'Yds', 'FL']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Add contextual features
    data = add_contextual_features(data)

    # Filter by position and add position-specific features
    pos_data = data[data['FantPos'] == position].copy()
    pos_data = add_advanced_position_features(pos_data, position)

    # Enhanced feature set
    base_features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Tgt', 'Rec', 'Y/R', 'Y/A', 'Fmb', 'FL']

    contextual_features = [
        'Team_Change', 'Games_Missed_Prev', 'Injury_Recovery',
        'Low_Usage_High_Potential', 'High_Prev_Performance',
        'PPR_Std', 'Seasons_Played'
    ]

    # Position-specific features
    if position == 'QB':
        pos_features = ['Pass_Att_PG', 'QB_Efficiency', 'Turnover_Rate', 'Low_Attempts_Prev', 'QB_Streaming_Penalty']
    elif position == 'RB':
        pos_features = ['Snap_Share_Proxy', 'Goal_Line_Upside', 'Young_Opportunity', 'Workload_Bonus', 'Elite_Workload_Bonus', 'Proven_Starter_Bonus', 'Committee_Risk']
    else:
        pos_features = ['Target_Efficiency', 'Red_Zone_Value', 'Target_Regression_Protection', 'Proven_Target_Bonus']

    all_features = base_features + contextual_features + pos_features

    # Prepare feature matrix
    feature_cols = [col for col in all_features if col in pos_data.columns]
    features_df = pos_data[feature_cols].copy()

    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    print(f"{position}: Using {len(feature_cols)} features including contextual factors")
    print(f"Key new features: {contextual_features[:3]}")

    return features_df, pos_data['PPR_Points']

if __name__ == "__main__":
    # Test the enhanced features
    for pos in ['QB', 'RB', 'WR', 'TE']:
        try:
            X, y = build_enhanced_model(pos)
            print(f"{pos}: {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"{pos}: Error - {e}")
