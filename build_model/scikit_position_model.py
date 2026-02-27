import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt

def build_position_model(position):
    # Load and process data
    import sys
    sys.path.append('..')
    from data_cleaning.create_df import all_data
    from team_context_integration import add_team_context_features, get_team_context_features
    from enhanced_features import add_contextual_features, add_advanced_position_features

    data = all_data.copy()

    # Add team context features
    data = add_team_context_features(data, year=2025)

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

    # Use enhanced_features for all contextual features
    data = add_contextual_features(data_sorted)

    # Filter by position
    pos_data = data[data['FantPos'] == position].copy()

    if len(pos_data) < 50:
        print(f"Skipping {position}: only {len(pos_data)} samples")
        return None

    # Use enhanced_features for position-specific features
    pos_data = add_advanced_position_features(pos_data, position)

    # Base features + contextual features from enhanced_features.py
    base_features = [
        'Age', 'G', 'GS',
        'Cmp', 'Att', 'Yds', 'TD', 'Int',
        'Tgt', 'Rec', 'Y/R', 'Y/A', 'Fmb', 'FL',
        'FantPt_Change', 'Games_Change'
    ]

    # Contextual features from enhanced_features.py
    contextual_features = [
        'Team_Change', 'Games_Missed_Prev', 'Injury_Recovery', 'Injury_Recovery_Boost',
        'Low_Usage_High_Potential', 'Total_Touches', 'Workload_Premium', 'Elite_Workload',
        'Proven_Starter', 'QB_Streaming_Penalty', 'Committee_Risk', 'Proven_Performer',
        'Elite_Target_Share', 'Target_Regression_Risk', 'PPR_Std', 'Seasons_Played',
        'Experience', 'Is_Rookie', 'Is_Sophomore'
    ]

    # Position-specific features from enhanced_features.py
    if position == 'QB':
        pos_features = ['Pass_Att_PG', 'QB_Efficiency', 'Turnover_Rate', 'Low_Attempts_Prev',
                       'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_YPG', 'Rush_Att_PG', 'QB_Mobility_Score']
    elif position == 'RB':
        pos_features = ['Snap_Share_Proxy', 'Goal_Line_Upside', 'Workload_Bonus', 'Elite_Workload_Bonus',
                       'Proven_Starter_Bonus', 'Young_Opportunity', 'Touches_PG', 'Target_Share',
                       'RB_Age_Penalty', 'High_Workload']
    else:  # WR/TE
        pos_features = ['Target_Efficiency', 'Red_Zone_Value', 'Target_Regression_Protection',
                       'Proven_Target_Bonus', 'Target_PG', 'Catch_Rate', 'YPG']

    # Add team context features for this position
    team_features = get_team_context_features(position)

    # Combine all features
    all_features = base_features + contextual_features + pos_features + team_features
    feature_cols = [col for col in all_features if col in pos_data.columns]

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
