import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

sys.path.append('../build_model')
sys.path.append('..')
from model_core import build_prediction_features
from data_cleaning.create_df import load_fantasy_stats

# Prediction year, passed on the command line: python predict_position_specific.py 2026
PREDICTION_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

def predict_with_position_models(year=PREDICTION_YEAR):
    # Load the most recently completed season's stats - this is what actually
    # gets fed into the model as features (see model_core.build_lagged_dataset
    # for why a player's own season stats can't be used to predict that same
    # season's points).
    latest_data = load_fantasy_stats(f'../stats/fantasy_stats_for_{year-1}.csv')

    try:
        prev_data = load_fantasy_stats(f'../stats/fantasy_stats_for_{year-2}.csv')
    except FileNotFoundError:
        prev_data = None

    predictions = []

    for position in ['QB', 'RB', 'WR', 'TE']:
        # Load position model
        try:
            with open(f'../models/{year}/model_{position.lower()}.pkl', 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                model_features = model_data['feature_names']
        except FileNotFoundError:
            print(f"Model for {position} not found, skipping...")
            continue

        pos_data, features_df = build_prediction_features(
            latest_data, year, position, model_features, prev_source_data=prev_data
        )

        if features_df is None or len(pos_data) == 0:
            continue

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
    os.makedirs(f'../predictions/{year}', exist_ok=True)
    pred_df.to_csv(f'../predictions/{year}/fantasy_predictions_position_specific_{year}.csv', index=False)
    print(f"\\nPosition-specific predictions saved for {year}")

    return pred_df

# Generate predictions
print("Generating position-specific predictions...")
print()

predictions = predict_with_position_models(PREDICTION_YEAR)
print(f"Generated {len(predictions)} predictions")
print("\\nTop 10 predictions:")
print(predictions.head(10)[['Player', 'Position', 'Predicted_Fantasy_Points', 'Rank']])
