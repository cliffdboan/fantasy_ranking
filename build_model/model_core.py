"""
Shared model-building logic used by scikit_position_model.py (production
training), predicting/predict_position_specific.py (production inference),
and backtest.py (walk-forward evaluation against past seasons).

Centralizing this is what lets the backtest harness score the exact same
feature-construction path that real predictions use - otherwise a backtest
number wouldn't actually tell you anything about real-world accuracy.

No argv parsing and no top-level side effects here on purpose, so importing
this module is always safe regardless of which CLI entry point does it.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from team_context_integration import add_team_context_features, get_team_context_features
from enhanced_features import add_contextual_features, add_advanced_position_features


def build_lagged_dataset(raw_data):
    """Reshape one-row-per-player-season data into forecast pairs: season S's
    stats as features, that SAME player's season S+1 PPR as the target.

    The season-S box score (Att, Yds, TD, Rec, ...) is also what PPR points
    are computed FROM, so training on season-S stats -> season-S PPR mostly
    just re-derives the scoring formula rather than forecasting anything.
    Predicting a player's NEXT season from their most recent one is what the
    model actually needs to do, since that's exactly the situation at
    inference time (predict_position_specific.py feeds in last season's
    stats to forecast the upcoming season).
    """
    data = raw_data.copy()
    data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')
    data['PPR_Points'] = pd.to_numeric(data['PPR'], errors='coerce')
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['G'] = pd.to_numeric(data['G'], errors='coerce')
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

    data = data.sort_values(['Player', 'Year'])
    by_player = data.groupby('Player')

    # Forward-looking: what we're actually trying to forecast.
    data['Target_PPR'] = by_player['PPR_Points'].shift(-1)
    data['Target_Year'] = by_player['Year'].shift(-1)

    # Backward-looking trend features describing the player's trajectory INTO
    # the feature season.
    prev_stats = by_player[['PPR_Points', 'G']].shift(1)
    data['FantPt_Change'] = data['PPR_Points'] - prev_stats['PPR_Points']
    data['Games_Change'] = data['G'] - prev_stats['G']

    # Keep only true season-over-season pairs, so a gap year (injury,
    # retirement, missing data) isn't treated as a normal transition.
    data = data[data['Target_Year'] == data['Year'] + 1].copy()

    return data


def add_team_context_by_target_year(data, year_col='Target_Year'):
    """Merge team context using EACH ROW's own target season, instead of one
    hardcoded year applied across the whole multi-decade dataset. Team data
    only exists for a couple of recent years (team_data/{year}/); rows whose
    target season has no matching folder simply get no team-context columns
    (add_team_context_features falls back to returning the chunk unchanged),
    the same as any other engineered feature would if it were unavailable -
    they get filled with 0 alongside everything else downstream.
    """
    chunks = []
    missing_years = []
    for year, chunk in data.groupby(year_col):
        year = int(year)
        before_cols = set(chunk.columns)
        merged_chunk = add_team_context_features(chunk, year=year, verbose=False)
        if set(merged_chunk.columns) == before_cols:
            missing_years.append(year)
        chunks.append(merged_chunk)

    if missing_years:
        print(f"No team_data/ found for {len(missing_years)} target year(s) "
              f"({min(missing_years)}-{max(missing_years)}); those rows get no team-context features.")

    return pd.concat(chunks, ignore_index=True) if chunks else data


def get_feature_columns(position):
    """Feature list for a position model. Shared by training and inference so
    the two can never silently drift apart."""
    base_features = [
        'Age', 'G', 'GS',
        'Cmp', 'Att', 'Yds', 'TD', 'Int',
        'Tgt', 'Rec', 'Y/R', 'Y/A', 'Fmb', 'FL',
        'FantPt_Change', 'Games_Change'
    ]

    contextual_features = [
        'Team_Change', 'Games_Missed_Prev', 'Injury_Recovery', 'Injury_Recovery_Boost',
        'Low_Usage_High_Potential', 'Total_Touches', 'Workload_Premium', 'Elite_Workload',
        'Proven_Starter', 'QB_Streaming_Penalty', 'Committee_Risk', 'Proven_Performer',
        'Elite_Target_Share', 'Target_Regression_Risk', 'PPR_Std', 'Seasons_Played',
        'Experience', 'Is_Rookie', 'Is_Sophomore',
        'PPR_Weighted_3yr', 'PPR_Trend_Slope', 'Touches_Weighted_3yr', 'Touches_Trend_Slope',
        'Recent_PPR_Std', 'Games_Missed_2yr', 'Tgt_Share', 'Tgt_Share_Trend',
        'Rush_Share', 'Rush_Share_Trend'
    ]

    if position == 'QB':
        pos_features = ['Pass_Att_PG', 'QB_Efficiency', 'Turnover_Rate', 'Low_Attempts_Prev',
                       'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_YPG', 'Rush_Att_PG', 'QB_Mobility_Score',
                       'QB_Age_Penalty']
    elif position == 'RB':
        pos_features = ['Snap_Share_Proxy', 'Goal_Line_Upside', 'Workload_Bonus', 'Elite_Workload_Bonus',
                       'Proven_Starter_Bonus', 'Young_Opportunity', 'Touches_PG', 'Target_Share',
                       'RB_Age_Penalty', 'High_Workload', 'RB_Age_Cliff']
    else:  # WR/TE
        pos_features = ['Target_Efficiency', 'Red_Zone_Value', 'Target_Regression_Protection',
                       'Proven_Target_Bonus', 'Target_PG', 'Catch_Rate', 'YPG', 'WR_TE_Age_Penalty']

    team_features = get_team_context_features(position)

    return base_features + contextual_features + pos_features + team_features


def make_model(position):
    """Position-tuned RandomForest (same hyperparameters this project already
    used - untouched by the train/target fix)."""
    if position == 'QB':
        return RandomForestRegressor(n_estimators=125, max_depth=10, random_state=42)
    elif position == 'RB':
        return RandomForestRegressor(n_estimators=175, max_depth=12, random_state=42)
    else:
        return RandomForestRegressor(n_estimators=150, max_depth=11, random_state=42)


def fit_position_model(lagged_data, position, as_of_year=None):
    """Fit a position model on the lagged (season S -> season S+1) dataset.

    as_of_year, when given, restricts training to rows whose TARGET season is
    strictly before it, so a walk-forward backtest fold never trains on
    information from the year it's about to predict.

    Returns None if there isn't enough data for this position, otherwise a
    dict with the fitted model, its feature columns, held-out metrics, and
    feature importances.
    """
    data = lagged_data
    if as_of_year is not None:
        data = data[data['Target_Year'] < as_of_year]

    data = add_team_context_by_target_year(data)
    data = add_contextual_features(data)

    pos_data = data[data['FantPos'] == position].copy()
    if len(pos_data) < 50:
        print(f"Skipping {position}: only {len(pos_data)} samples")
        return None

    pos_data = add_advanced_position_features(pos_data, position)

    feature_cols = [col for col in get_feature_columns(position) if col in pos_data.columns]
    features_df = pos_data[feature_cols].copy()
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    X = features_df.values
    y = pos_data['Target_PPR'].values

    valid_idx = ~pd.isna(y)
    X = X[valid_idx]
    y = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_model(position)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'mae': mae,
        'r2': r2,
        'samples': len(pos_data),
        'y_test': y_test,
        'y_pred': y_pred,
    }


def build_prediction_features(source_data, target_year, position, model_features, prev_source_data=None):
    """Build the feature matrix used to forecast target_year for one
    position, from source_data (one row per player for target_year - 1).

    prev_source_data (target_year - 2), when given, is used only to compute
    trend features (FantPt_Change, Games_Change) the same way training's
    lag does; without it those default to 0, same as today's behavior when
    no such file is found.

    Returns (pos_data, features_df) - pos_data carries player/team/age
    metadata for building the output rows, features_df is the numeric
    matrix aligned to model_features and ready for model.predict().
    """
    data = source_data.copy()
    data = add_team_context_features(data, year=target_year)
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(25)

    if prev_source_data is not None and not prev_source_data.empty:
        prev_stats = prev_source_data.drop_duplicates('Player').set_index('Player')[['FantPt', 'G']]
        data['Prev_FantPt'] = pd.to_numeric(data['Player'].map(prev_stats['FantPt']), errors='coerce').fillna(0)
        data['FantPt_Change'] = pd.to_numeric(data['FantPt'], errors='coerce') - data['Prev_FantPt']
        data['Games_Change'] = pd.to_numeric(data['G'], errors='coerce') - pd.to_numeric(
            data['Player'].map(prev_stats['G']), errors='coerce').fillna(17)
    else:
        data['FantPt_Change'] = 0
        data['Games_Change'] = 0

    data = add_contextual_features(data)

    pos_data = data[data['FantPos'] == position].copy()
    if len(pos_data) == 0:
        return pos_data, None

    pos_data['G'] = pd.to_numeric(pos_data['G'], errors='coerce').fillna(1)
    pos_data = add_advanced_position_features(pos_data, position)

    features_df = pd.DataFrame()
    for feature in model_features:
        if feature in pos_data.columns:
            features_df[feature] = pd.to_numeric(pos_data[feature], errors='coerce').fillna(0)
        else:
            features_df[feature] = 0

    return pos_data, features_df
