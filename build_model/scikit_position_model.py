import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

def build_improved_position_model(position):
    # Load and process data
    import sys
    sys.path.append('..')
    from data_cleaning.create_df import all_data
    from team_context_integration import add_team_context_features, get_team_context_features

    data = all_data.copy()

    # Add team context features
    data = add_team_context_features(data, year=2024)

    # Use PPR scoring directly from data
    data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')
    data['PPR_Points'] = pd.to_numeric(data['PPR'], errors='coerce')
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['G'] = pd.to_numeric(data['G'], errors='coerce')

    # Add trend features using PPR scoring
    data_sorted = data.sort_values(['Player', 'Age'])
    prev_stats = data_sorted.groupby('Player')[['PPR_Points', 'G']].shift(1)
    data_sorted['FantPt_Change'] = data_sorted['PPR_Points'] - prev_stats['PPR_Points']
    data_sorted['Games_Change'] = data_sorted['G'] - prev_stats['G']

    # Add contextual features that explain major prediction errors
    # Team change penalty
    data_sorted['Team_Change'] = data_sorted.groupby('Player')['Tm'].transform(lambda x: (x != x.shift(1)).astype(int))

    # Injury recovery indicator
    prev_games = pd.to_numeric(prev_stats['G'], errors='coerce')
    data_sorted['Games_Missed_Prev'] = 17 - prev_games
    data_sorted['Injury_Recovery'] = np.where(data_sorted['Games_Missed_Prev'] > 8, 1, 0)

    # Low usage/high potential
    prev_ppr = pd.to_numeric(data_sorted.groupby('Player')['PPR_Points'].shift(1), errors='coerce')
    data_sorted['Low_Usage_High_Potential'] = np.where((prev_ppr < 100) & (data_sorted['Age'] < 26), 1, 0)

    data = data_sorted.fillna(0)

    # Filter by position
    pos_data = data[data['FantPos'] == position].copy()

    if len(pos_data) < 50:
        print(f"Skipping {position}: only {len(pos_data)} samples")
        return None

    # Start with original features + new contextual features
    base_features = [
        'Age', 'G', 'GS',
        'Cmp', 'Att', 'Yds', 'TD', 'Int',
        'Tgt', 'Rec', 'Y/R',
        'Y/A',
        'Fmb', 'FL',
        'FantPt_Change', 'Games_Change',
        'Team_Change', 'Games_Missed_Prev', 'Injury_Recovery', 'Low_Usage_High_Potential'
    ]

    # Add position-specific enhanced features
    enhanced_features = []

    if position == 'QB':
        # QB rushing upside
        pos_data['Rush_Att'] = pd.to_numeric(pos_data['Att.1'], errors='coerce').fillna(0)
        pos_data['Rush_Yds'] = pd.to_numeric(pos_data['Yds.1'], errors='coerce').fillna(0)
        pos_data['Rush_TD'] = pd.to_numeric(pos_data['TD.1'], errors='coerce').fillna(0)

        # Key QB rushing metrics
        pos_data['Rush_YPG'] = pos_data['Rush_Yds'] / pos_data['G'].replace(0, 1)
        pos_data['Rush_Att_PG'] = pos_data['Rush_Att'] / pos_data['G'].replace(0, 1)
        pos_data['QB_Mobility_Score'] = pos_data['Rush_YPG'] + (pos_data['Rush_Att_PG'] * 2)  # Weight attempts higher

        # Breakout QB detection
        prev_att = pd.to_numeric(pos_data.groupby('Player')['Att'].shift(1), errors='coerce')
        pos_data['Low_Attempts_Prev'] = np.where(prev_att < 200, 1, 0)

        enhanced_features = ['Rush_YPG', 'Rush_Att_PG', 'QB_Mobility_Score', 'Low_Attempts_Prev']

    elif position == 'RB':
        # RB improvements: Better workload prediction (highest MAE issue)
        pos_data['Total_Touches'] = pd.to_numeric(pos_data['Att.1'], errors='coerce').fillna(0) + pd.to_numeric(pos_data['Rec'], errors='coerce').fillna(0)
        pos_data['Touches_PG'] = pos_data['Total_Touches'] / pos_data['G'].replace(0, 1)
        pos_data['Target_Share'] = pd.to_numeric(pos_data['Tgt'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)

        # Age penalty for RBs (sharp decline after 27)
        pos_data['RB_Age_Penalty'] = np.where(pos_data['Age'] > 27, (pos_data['Age'] - 27) ** 1.5, 0)

        # Workload sustainability
        pos_data['High_Workload'] = np.where(pos_data['Touches_PG'] > 18, 1, 0)

        enhanced_features = ['Total_Touches', 'Touches_PG', 'Target_Share', 'RB_Age_Penalty', 'High_Workload']

    elif position in ['WR', 'TE']:
        # WR/TE improvements: Target quality metrics
        pos_data['Target_PG'] = pd.to_numeric(pos_data['Tgt'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)
        pos_data['Catch_Rate'] = pd.to_numeric(pos_data['Rec'], errors='coerce').fillna(0) / pd.to_numeric(pos_data['Tgt'], errors='coerce').replace(0, 1)
        pos_data['YPG'] = pd.to_numeric(pos_data['Yds.2'], errors='coerce').fillna(0) / pos_data['G'].replace(0, 1)

        enhanced_features = ['Target_PG', 'Catch_Rate', 'YPG']

    # Add experience tracking (rookie penalty for overvaluation)
    pos_data_sorted = pos_data.sort_values(['Player', 'Age'])
    pos_data_sorted['Experience'] = pos_data_sorted.groupby('Player').cumcount() + 1
    pos_data_sorted['Is_Rookie'] = np.where(pos_data_sorted['Experience'] == 1, 1, 0)
    pos_data_sorted['Is_Sophomore'] = np.where(pos_data_sorted['Experience'] == 2, 1, 0)

    # Update pos_data with sorted version
    pos_data = pos_data_sorted.copy()

    enhanced_features.extend(['Experience', 'Is_Rookie', 'Is_Sophomore'])

    # Add team context features for this position
    team_features = get_team_context_features(position)

    # Combine all features
    all_features = base_features + enhanced_features + team_features
    feature_cols = [col for col in all_features if col in pos_data.columns]

    # Add enhanced features to dataframe
    for feat in enhanced_features:
        if feat not in pos_data.columns:
            print(f"Warning: {feat} not found in data for {position}")

    features_df = pos_data[feature_cols].copy()

    # Convert to numeric and fill NaNs
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df[col] = features_df[col].fillna(0)

    X = features_df.values
    y = pos_data['PPR_Points'].values

    # Remove rows with NaN targets
    valid_idx = ~pd.isna(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Enhanced model parameters based on position characteristics
    if position == 'QB':
        # QBs more predictable, fewer trees needed
        model = RandomForestRegressor(n_estimators=125, max_depth=10, random_state=42)
    elif position == 'RB':
        # RBs high variance, need more complex model
        model = RandomForestRegressor(n_estimators=175, max_depth=12, random_state=42)
    else:
        # WR/TE standard enhanced parameters
        model = RandomForestRegressor(n_estimators=150, max_depth=11, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{position}: MAE = {mae:.1f}, R² = {r2:.3f}, Samples = {len(pos_data)}, Features = {len(feature_cols)}")

    # Show feature importance for top features
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Top 5 features for {position}:")
    for i, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # Create visualizations
    create_position_visuals(position, feature_importance, y_test, y_pred, mae, r2)

    # Save model and feature names
    model_data = {
        'model': model,
        'feature_names': feature_cols,
        'feature_importance': feature_importance
    }
    # Save to models directory
    with open(f'../models/model_{position.lower()}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    return model, {'mae': mae, 'r2': r2, 'samples': len(pos_data)}

def create_position_visuals(position, feature_importance, y_test, y_pred, mae, r2):
    """Create visualizations for position model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{position} Model Analysis', fontsize=16)

    # Feature importance
    top_features = feature_importance.head(10)
    axes[0,0].barh(range(len(top_features)), top_features['importance'])
    axes[0,0].set_yticks(range(len(top_features)))
    axes[0,0].set_yticklabels(top_features['feature'])
    axes[0,0].set_title('Top 10 Feature Importance')
    axes[0,0].invert_yaxis()

    # Prediction vs Actual scatter
    axes[0,1].scatter(y_test, y_pred, alpha=0.6)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0,1].set_xlabel('Actual Fantasy Points')
    axes[0,1].set_ylabel('Predicted Fantasy Points')
    axes[0,1].set_title(f'Predictions vs Actual\nMAE: {mae:.1f}, R²: {r2:.3f}')

    # Residuals
    residuals = y_pred - y_test
    axes[1,0].scatter(y_pred, residuals, alpha=0.6)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Fantasy Points')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residual Plot')

    # Error distribution
    axes[1,1].hist(np.abs(residuals), bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Absolute Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution')

    plt.tight_layout()
    plt.savefig(f'../model_analysis/{position}_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visual(model_stats):
    """Create summary comparison across positions"""
    positions = list(model_stats.keys())
    maes = [model_stats[pos]['mae'] for pos in positions]
    r2s = [model_stats[pos]['r2'] for pos in positions]
    samples = [model_stats[pos]['samples'] for pos in positions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Summary', fontsize=16)

    # MAE comparison
    axes[0].bar(positions, maes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('MAE by Position')
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center')

    # R² comparison
    axes[1].bar(positions, r2s, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² by Position')
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')

    # Sample sizes
    axes[2].bar(positions, samples, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[2].set_ylabel('Training Samples')
    axes[2].set_title('Sample Size by Position')
    for i, v in enumerate(samples):
        axes[2].text(i, v + 10, f'{v}', ha='center')

    plt.tight_layout()
    plt.savefig('../model_analysis/model_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# Build improved models for each position
positions = ['QB', 'RB', 'WR', 'TE']
models = {}
model_stats = {}

print("Building position-specific models...")
print()

for pos in positions:
    model, stats = build_improved_position_model(pos)
    models[pos] = model
    model_stats[pos] = stats

# Create summary visualization
create_summary_visual(model_stats)

print("\nPosition-specific models created!")
print("Models saved in /models/ directory as: model_qb.pkl, model_rb.pkl, etc.")
print("Visualizations saved in model_analysis/ directory")
