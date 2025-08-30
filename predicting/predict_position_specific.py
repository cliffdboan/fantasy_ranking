import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# while True:
#     try:
#         PREDICTION_YEAR = int(input("Enter the year to predict: "))
#         if PREDICTION_YEAR < 2000 or PREDICTION_YEAR > datetime.now().year:
#             raise ValueError
#         break
#     except ValueError:
#         print(f"Invalid year. Please enter a year between 2000 and {datetime.now().year}.")
PREDICTION_YEAR = 2025

def predict_with_position_models(year=PREDICTION_YEAR):
    # Load current year's stats (same pattern as existing script)
    latest_data = pd.read_csv(f'../stats/fantasy_stats_for_{year-1}.csv', header=1)

    # Add team context features
    import sys
    sys.path.append('../build_model')
    from team_context_integration import add_team_context_features
    latest_data = add_team_context_features(latest_data, year=year)
    try:
        prev_data = pd.read_csv(f'../stats/fantasy_stats_for_{year-2}.csv', header=1)
        prev_stats = prev_data.drop_duplicates('Player').set_index('Player')[['FantPt', 'G']]
    except:
        prev_stats = pd.DataFrame()

    # Add trend features (same as existing)
    if not prev_stats.empty:
        latest_data['Prev_FantPt'] = pd.to_numeric(latest_data['Player'].map(prev_stats['FantPt']), errors='coerce').fillna(0)
        latest_data['FantPt_Change'] = pd.to_numeric(latest_data['FantPt'], errors='coerce') - latest_data['Prev_FantPt']
        latest_data['Games_Change'] = pd.to_numeric(latest_data['G'], errors='coerce') - pd.to_numeric(latest_data['Player'].map(prev_stats['G']), errors='coerce').fillna(17)
    else:
        latest_data['FantPt_Change'] = 0
        latest_data['Games_Change'] = 0

    # NEW: Add contextual features for prediction
    # Team change (assume 0 for prediction year - would need roster data to detect)
    latest_data['Team_Change'] = 0

    # Games missed previous year (injury recovery)
    if not prev_stats.empty:
        prev_games = pd.to_numeric(latest_data['Player'].map(prev_stats['G']), errors='coerce').fillna(17)
        latest_data['Games_Missed_Prev'] = 17 - prev_games
        latest_data['Injury_Recovery'] = np.where(latest_data['Games_Missed_Prev'] > 8, 1, 0)
    else:
        latest_data['Games_Missed_Prev'] = 0
        latest_data['Injury_Recovery'] = 0

    # Low usage/high potential
    if not prev_stats.empty:
        prev_ppr = pd.to_numeric(latest_data['Player'].map(prev_stats['FantPt']), errors='coerce').fillna(0)
        latest_data['Low_Usage_High_Potential'] = np.where(
            (prev_ppr < 100) & (pd.to_numeric(latest_data['Age'], errors='coerce') < 26), 1, 0
        )
    else:
        latest_data['Low_Usage_High_Potential'] = 0

    predictions = []

    # Base features (same as existing)
    base_performance_stats = [
        'Age', 'G', 'GS',
        'Cmp', 'Att', 'Yds', 'TD', 'Int',
        'Tgt', 'Rec', 'Y/R',
        'Y/A',
        'Fmb', 'FL',
        'FantPt_Change', 'Games_Change'
    ]

    for position in ['QB', 'RB', 'WR', 'TE']:
        # Load position model
        try:
            with open(f'../models/model_{position.lower()}.pkl', 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                model_features = model_data['feature_names']
        except FileNotFoundError:
            print(f"Model for {position} not found, skipping...")
            continue

        # Get position data
        pos_data = latest_data[latest_data['FantPos'] == position].copy()

        if len(pos_data) == 0:
            continue

        # Convert key columns to numeric first
        pos_data['G'] = pd.to_numeric(pos_data['G'], errors='coerce').fillna(1)
        pos_data['Age'] = pd.to_numeric(pos_data['Age'], errors='coerce').fillna(25)

        # Add enhanced features based on position
        if position == 'QB':
            pos_data['Rush_Att'] = pd.to_numeric(pos_data['Att.1'], errors='coerce').fillna(0)
            pos_data['Rush_Yds'] = pd.to_numeric(pos_data['Yds.1'], errors='coerce').fillna(0)
            pos_data['Rush_TD'] = pd.to_numeric(pos_data['TD.1'], errors='coerce').fillna(0)
            pos_data['Rush_YPG'] = pos_data['Rush_Yds'] / pos_data['G'].replace(0, 1)
            pos_data['Rush_Att_PG'] = pos_data['Rush_Att'] / pos_data['G'].replace(0, 1)
            pos_data['QB_Mobility_Score'] = pos_data['Rush_YPG'] + (pos_data['Rush_Att_PG'] * 2)

            # NEW: Low attempts previous year (breakout detection)
            if not prev_stats.empty:
                prev_att = pos_data['Player'].map(prev_stats.get('Att', pd.Series())).fillna(0)
                pos_data['Low_Attempts_Prev'] = np.where(prev_att < 200, 1, 0)
            else:
                pos_data['Low_Attempts_Prev'] = 0

        elif position == 'RB':
            pos_data['Total_Touches'] = pd.to_numeric(pos_data['Att.1'], errors='coerce').fillna(0) + pd.to_numeric(pos_data['Rec'], errors='coerce').fillna(0)
            pos_data['Touches_PG'] = pos_data['Total_Touches'] / pos_data['G'].replace(0, 1)
            pos_data['Target_Share'] = pd.to_numeric(pos_data['Tgt'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)
            pos_data['RB_Age_Penalty'] = np.where(pos_data['Age'] > 27, (pos_data['Age'] - 27) ** 1.5, 0)
            pos_data['High_Workload'] = np.where(pos_data['Touches_PG'] > 18, 1, 0)

        elif position in ['WR', 'TE']:
            pos_data['Target_PG'] = pd.to_numeric(pos_data['Tgt'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)
            pos_data['Catch_Rate'] = pd.to_numeric(pos_data['Rec'], errors='coerce').fillna(0) / pd.to_numeric(pos_data['Tgt'], errors='coerce').replace(0, 1)
            pos_data['YPG'] = pd.to_numeric(pos_data['Yds.2'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)

        # Add experience features (rookie/sophomore adjustments)
        pos_data_sorted = pos_data.sort_values(['Player', 'Age'])
        pos_data_sorted['Experience'] = pos_data_sorted.groupby('Player').cumcount() + 1
        pos_data_sorted['Is_Rookie'] = np.where(pos_data_sorted['Experience'] == 1, 1, 0)
        pos_data_sorted['Is_Sophomore'] = np.where(pos_data_sorted['Experience'] == 2, 1, 0)
        pos_data = pos_data_sorted.copy()

        # Prepare features - align with model expectations
        features_df = pd.DataFrame()
        for feature in model_features:
            if feature in pos_data.columns:
                features_df[feature] = pd.to_numeric(pos_data[feature], errors='coerce').fillna(0)
            else:
                features_df[feature] = 0

        # Make predictions
        pred_points = model.predict(features_df.values)

        # Create results (same format as existing)
        for i, (idx, row) in enumerate(pos_data.iterrows()):
            predictions.append({
                'Player': row['Player'],
                'Position': position,
                'Team': row.get('Tm', 'UNK'),
                'Age': row.get('Age', 0),
                'Last_Season_Points': row.get('FantPt', 0),
                'Predicted_Fantasy_Points': pred_points[i]
            })

        print(f"{position}: {len(pos_data)} predictions made")

    # Convert to DataFrame and rank (same as existing)
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('Predicted_Fantasy_Points', ascending=False)
    pred_df['Rank'] = range(1, len(pred_df) + 1)

    # Save predictions
    pred_df.to_csv(f'../predictions/fantasy_predictions_position_specific_{year}.csv', index=False)
    print(f"\\nPosition-specific predictions saved for {year}")

    return pred_df

# Generate predictions
print("Generating position-specific predictions...")
print()

predictions = predict_with_position_models(PREDICTION_YEAR)
print(f"Generated {len(predictions)} predictions")
print("\\nTop 10 predictions:")
print(predictions.head(10)[['Player', 'Position', 'Predicted_Fantasy_Points', 'Rank']])
