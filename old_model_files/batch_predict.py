import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from data_cleaning.create_df import all_data

CURRENT_YEAR = 2023

def load_model_components():
    """Load model and recreate scaler/feature_names from training data"""
    # Load saved model
    model = tf.keras.models.load_model('fantasy_football_model.keras')

    # Recreate scaler and feature names (same process as build_tensor.py)
    data = all_data.copy()
    data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')
    data['G'] = pd.to_numeric(data['G'], errors='coerce')
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # Add trend features (same as training)
    data_sorted = data.sort_values(['Player', 'Age'])
    prev_stats = data_sorted.groupby('Player')[['FantPt', 'G']].shift(1)
    data_sorted['FantPt_Change'] = data_sorted['FantPt'] - prev_stats['FantPt']
    data_sorted['Games_Change'] = data_sorted['G'] - prev_stats['G']
    data = data_sorted.fillna(0)

    mask = data['FantPt'].notna()
    data = data[mask].copy().reset_index(drop=True)
    data.pop('FantPt')  # Remove target

    # Use raw stats only - NO fantasy scoring columns
    performance_stats = [
        'Age', 'G', 'GS',  # Basic info
        'Cmp', 'Att', 'Yds', 'TD', 'Int',  # Passing stats
        'Tgt', 'Rec', 'Y/R',  # Receiving stats
        'Y/A',  # Rushing efficiency
        'Fmb', 'FL',  # Turnovers
        'FantPt_Change', 'Games_Change'  # Trend features
    ]

    # Select only performance columns that exist
    feature_cols = []
    for col in performance_stats:
        if col in data.columns:
            feature_cols.append(col)

    # Create features dataframe with only performance stats
    features_df = data[feature_cols].copy()

    # Convert all to numeric and fill NaNs
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df[col] = features_df[col].fillna(0)
    scaler = StandardScaler()
    scaler.fit(features_df.values)

    return model, scaler, features_df.columns.tolist()

def predict_all_players():
    """Generate predictions for all players and export results"""

    # Load model and preprocessing components
    model, scaler, feature_names = load_model_components()

    # Load current year's stats and previous year for trend analysis
    latest_data = pd.read_csv(f'stats/fantasy_stats_for_{CURRENT_YEAR}.csv')
    try:
        prev_data = pd.read_csv(f'stats/fantasy_stats_for_{CURRENT_YEAR-1}.csv')
        prev_stats = prev_data.drop_duplicates('Player').set_index('Player')[['FantPt', 'G']]
    except:
        prev_stats = pd.DataFrame()

    print(f"Making predictions for {len(latest_data)} players from {CURRENT_YEAR} season")

    # Add trend features if previous year data available
    if not prev_stats.empty:
        latest_data['Prev_FantPt'] = latest_data['Player'].map(prev_stats['FantPt']).fillna(0)
        latest_data['FantPt_Change'] = pd.to_numeric(latest_data['FantPt'], errors='coerce') - latest_data['Prev_FantPt']
        latest_data['Games_Change'] = latest_data['G'] - latest_data['Player'].map(prev_stats['G']).fillna(17)
    else:
        latest_data['FantPt_Change'] = 0
        latest_data['Games_Change'] = 0

    # Use raw stats only - NO fantasy scoring columns
    performance_stats = [
        'Age', 'G', 'GS',  # Basic info
        'Cmp', 'Att', 'Yds', 'TD', 'Int',  # Passing stats
        'Tgt', 'Rec', 'Y/R',  # Receiving stats
        'Y/A',  # Rushing efficiency
        'Fmb', 'FL',  # Turnovers
        'FantPt_Change', 'Games_Change'  # Trend features
    ]

    # Select only performance columns that exist
    feature_cols = []
    for col in performance_stats:
        if col in latest_data.columns:
            feature_cols.append(col)

    print(f"Available columns in latest_data: {list(latest_data.columns)}")
    print(f"Selected feature columns: {feature_cols}")

    # Create features dataframe with only performance stats
    features_df = latest_data[feature_cols].copy()

    # Convert all to numeric and fill NaNs
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df[col] = features_df[col].fillna(0)

    # Align with training features
    for feature in feature_names:
        if feature not in features_df.columns:
            features_df[feature] = 0
    features_df = features_df.reindex(columns=feature_names, fill_value=0)

    # Scale and predict
    X_scaled = scaler.transform(features_df.values)
    predictions = model.predict(X_scaled, verbose=0).flatten()

    # Create results DataFrame
    results = pd.DataFrame({
        'Player': latest_data['Player'],
        'Position': latest_data.get('FantPos', 'Unknown'),
        'Team': latest_data.get('Tm', 'Unknown'),
        'Age': latest_data.get('Age', 0),
        'Last_Season_Points': pd.to_numeric(latest_data.get('FantPt', 0), errors='coerce'),
        'Predicted_Fantasy_Points': predictions
    })

    # Sort by predicted points
    results = results.sort_values('Predicted_Fantasy_Points', ascending=False)
    results['Rank'] = range(1, len(results) + 1)

    # Export CSV
    results.to_csv(f'fantasy_predictions_{CURRENT_YEAR + 1}.csv', index=False)
    print(f"Exported predictions for {len(results)} players to 'fantasy_predictions_{CURRENT_YEAR + 1}.csv'")

    # Create visualization
    create_prediction_charts(results)

    return results

def create_prediction_charts(results):
    """Create visualization charts for predictions"""

    plt.figure(figsize=(16, 12))

    # Top 20 players
    plt.subplot(2, 3, 1)
    top_20 = results.head(20)
    plt.barh(range(len(top_20)), top_20['Predicted_Fantasy_Points'])
    plt.yticks(range(len(top_20)), top_20['Player'])
    plt.xlabel('Predicted Fantasy Points')
    plt.title('Top 20 Predicted Fantasy Performers')
    plt.gca().invert_yaxis()

    # By position
    plt.subplot(2, 3, 2)
    pos_avg = results.groupby('Position')['Predicted_Fantasy_Points'].mean().sort_values(ascending=False)
    pos_avg.plot(kind='bar')
    plt.title('Average Predicted Points by Position')
    plt.xticks(rotation=45)

    # Distribution
    plt.subplot(2, 3, 3)
    plt.hist(results['Predicted_Fantasy_Points'], bins=30, alpha=0.7)
    plt.xlabel('Predicted Fantasy Points')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Predicted Points')

    # Top QBs
    plt.subplot(2, 3, 4)
    qbs = results[results['Position'] == 'QB'].head(10)
    if len(qbs) > 0:
        plt.barh(range(len(qbs)), qbs['Predicted_Fantasy_Points'])
        plt.yticks(range(len(qbs)), qbs['Player'])
        plt.xlabel('Predicted Fantasy Points')
        plt.title('Top 10 QBs')
        plt.gca().invert_yaxis()

    # Top RBs
    plt.subplot(2, 3, 5)
    rbs = results[results['Position'] == 'RB'].head(10)
    if len(rbs) > 0:
        plt.barh(range(len(rbs)), rbs['Predicted_Fantasy_Points'])
        plt.yticks(range(len(rbs)), rbs['Player'])
        plt.xlabel('Predicted Fantasy Points')
        plt.title('Top 10 RBs')
        plt.gca().invert_yaxis()

    # Top WRs
    plt.subplot(2, 3, 6)
    wrs = results[results['Position'] == 'WR'].head(10)
    if len(wrs) > 0:
        plt.barh(range(len(wrs)), wrs['Predicted_Fantasy_Points'])
        plt.yticks(range(len(wrs)), wrs['Player'])
        plt.xlabel('Predicted Fantasy Points')
        plt.title('Top 10 WRs')
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'fantasy_predictions_{CURRENT_YEAR + 1}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved visualization as 'fantasy_predictions_{CURRENT_YEAR + 1}.png'")

if __name__ == "__main__":
    try:
        results = predict_all_players()
        print(f"\nTop 10 Predicted Fantasy Performers:")
        print(results[['Rank', 'Player', 'Position', 'Team', 'Last_Season_Points', 'Predicted_Fantasy_Points']].head(10))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've trained the model first by running: python model.py")
