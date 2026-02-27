import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# CHANGE THIS EACH YEAR THAT YOU WANT TO PREDICT
PREDICTION_YEAR = 2026

def predict_with_position_models(year=PREDICTION_YEAR):
    # Load current year's stats (same pattern as existing script)
    latest_data = pd.read_csv(f'../stats/fantasy_stats_for_{year-1}.csv', header=1)

    # Add team context features
    import sys
    sys.path.append('../build_model')
    from team_context_integration import add_team_context_features
    from enhanced_features import add_contextual_features, add_advanced_position_features

    latest_data = add_team_context_features(latest_data, year=year)

    # Add enhanced contextual features
    latest_data = add_contextual_features(latest_data)
    try:
        prev_data = pd.read_csv(f'../stats/fantasy_stats_for_{year-2}.csv', header=1)
        prev_stats = prev_data.drop_duplicates('Player').set_index('Player')[['FantPt', 'G']]
    except:
        prev_stats = pd.DataFrame()

    # Add trend features
    if not prev_stats.empty:
        latest_data['Prev_FantPt'] = pd.to_numeric(latest_data['Player'].map(prev_stats['FantPt']), errors='coerce').fillna(0)
        latest_data['FantPt_Change'] = pd.to_numeric(latest_data['FantPt'], errors='coerce') - latest_data['Prev_FantPt']
        latest_data['Games_Change'] = pd.to_numeric(latest_data['G'], errors='coerce') - pd.to_numeric(latest_data['Player'].map(prev_stats['G']), errors='coerce').fillna(17)
    else:
        latest_data['FantPt_Change'] = 0
        latest_data['Games_Change'] = 0

    predictions = []

    # Base features, used as human reference
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

        # Use enhanced position-specific features from enhanced_features.py
        pos_data = add_advanced_position_features(pos_data, position)

        # Prepare features - align with model expectations
        features_df = pd.DataFrame()
        for feature in model_features:
            if feature in pos_data.columns:
                features_df[feature] = pd.to_numeric(pos_data[feature], errors='coerce').fillna(0)
            else:
                features_df[feature] = 0

        # Make predictions
        pred_points = model.predict(features_df.values)

        # Create results
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

    # Convert to DataFrame and rank 
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
