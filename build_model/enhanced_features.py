import pandas as pd
import numpy as np
import os
import sys


def _clean_player_key(series):
    """Strip PFR's Pro-Bowl/All-Pro '*'/'+' markers, which get appended
    inconsistently year to year (a player can be 'Name' one season, 'Name*'
    the next, 'Name*+' the one after that). Left in place, a plain
    groupby('Player') silently splits one real player's career into several
    unrelated identities - which starves exactly the star players fantasy
    prediction cares most about of historical context. Mirrors the regex
    backtest.py already uses to clean names before scoring predictions."""
    return series.astype(str).str.replace(r'[*+]', '', regex=True).str.strip()


def _dedupe_player_seasons(df):
    """Collapse to one row per (player, season). A player traded mid-season
    gets both a combined row (Tm like '2TM') and partial per-team breakout
    rows in the source data; prefer the combined row since it reflects the
    player's true full-season total. Any other duplicate (rare: two real
    people who happen to share a cleaned name, e.g. two different players
    both named "Alex Smith") is broken by games played, just so the lookup
    below stays one-row-per-key and deterministic."""
    d = df.copy()
    d['_multi_team'] = d['Tm'].astype(str).str.match(r'^\dTM$').astype(int)
    d['_g_num'] = pd.to_numeric(d['G'], errors='coerce').fillna(0)
    d = d.sort_values(['_multi_team', '_g_num'], ascending=[False, False])
    d = d.drop_duplicates(subset=['_player_key', 'Year'], keep='first')
    return d.drop(columns=['_multi_team', '_g_num'])


def _team_season_totals(df):
    """Team-level rush attempts and targets per season, summed across every
    real roster row (any position - QB scrambles and WR jet sweeps count
    toward rushing volume too) for that team. This is the denominator for
    the opportunity-share features below. team_data/team_context_*.csv only
    covers 2024-2025 (a separate, concurrent workstream), far too thin for
    a model trained across 26 years, so this is derived directly from the
    historical player pool instead. Multi-team aggregate rows ('2TM', ...)
    are excluded so a traded player's stats aren't double-counted on top of
    their real per-team rows."""
    real = df[~df['Tm'].astype(str).str.match(r'^(\dTM|Tm)$')].copy()
    real['_rush_att'] = pd.to_numeric(real['Att.1'], errors='coerce').fillna(0)
    real['_tgt'] = pd.to_numeric(real['Tgt'], errors='coerce').fillna(0)
    return real.groupby(['Tm', 'Year']).agg(
        Team_Rush_Att=('_rush_att', 'sum'),
        Team_Tgt=('_tgt', 'sum'),
    ).reset_index()


def _lag_with_year_gate(player_key, year, values, lag):
    """`values` shifted `lag` seasons back within each player's own row
    sequence (already sorted by player, then year), kept only where the
    shifted row is truly `lag` calendar years earlier - so a gap season
    (injury, retirement, a missing stats file) never gets silently treated
    as if it were adjacent to the current one."""
    shifted_val = values.groupby(player_key).shift(lag)
    shifted_year = year.groupby(player_key).shift(lag)
    return shifted_val.where((year - shifted_year) == lag)


def _weighted_avg_3yr(v0, v1, v2):
    """Recency-weighted average of up to 3 seasons (weights 3/2/1, most
    recent first), renormalized over whichever of the prior seasons (v1, v2)
    actually exist - so a 2nd-year player's average leans on 2 real seasons
    instead of being dragged toward 0 by a missing 3rd."""
    w0, w1, w2 = 3.0, 2.0, 1.0
    num = v0.fillna(0) * w0 + v1.fillna(0) * w1 + v2.fillna(0) * w2
    den = w0 + np.where(v1.notna(), w1, 0.0) + np.where(v2.notna(), w2, 0.0)
    return num / den


def _trend_slope_3yr(v0, v1, v2):
    """OLS trend slope (points per season) over evenly-spaced season points,
    using whichever of the last up to 3 seasons are available: for 3 points
    the least-squares slope has the closed form (v0 - v2) / 2 (the middle
    point's residual always nets to zero against an evenly-spaced x, so only
    the endpoints matter); for 2 points it's the plain 1-year delta; with
    fewer than 2 there's no trend to measure, so it defaults to 0 - neutral,
    same convention every other feature here uses for "unknown"."""
    slope = pd.Series(0.0, index=v0.index)
    two_pt = v1.notna() & v2.isna()
    slope[two_pt] = (v0 - v1)[two_pt]
    three_pt = v1.notna() & v2.notna()
    slope[three_pt] = ((v0 - v2) / 2)[three_pt]
    return slope


def _build_multi_year_features():
    """One row per (player, season) of backward-looking, multi-season trend/
    opportunity-share/consistency features, built once from the complete
    player-season history in all_data (every year 2000-2025) rather than
    from whatever partial slice of rows a given caller passes in.

    That distinction matters beyond convenience: add_contextual_features is
    also called from build_prediction_features with a SINGLE season's
    snapshot (one row per player) for real inference and for backtest
    evaluation. A plain groupby('Player').shift(1) computed only from
    that `data` would always come back empty (a group of size 1 has no
    "previous" row) - which is exactly what already happens to several of
    the existing shift-based features here (Games_Missed_Prev, PPR_Std,
    Experience/Is_Rookie, Team_Change, ...): they look fine when trained on
    the multi-row lagged dataset, but silently collapse to near-constants
    at actual prediction/backtest time (e.g. ~97% of real veterans get
    flagged Is_Rookie there). Looking values up here instead, from the full
    history keyed by (player, season), gives the same answer whether the
    caller has 1 row or 30.

    Every value is built strictly from a season and the ones before it
    (shift/rolling look backward only), so merging this onto any row by
    that row's own (player, season) key can never reach past that row's
    feature season - the same lagged-forecasting rule build_lagged_dataset
    follows for the target itself.
    """
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from data_cleaning.create_df import all_data

    df = all_data.copy()
    df['_player_key'] = _clean_player_key(df['Player'])
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    team_totals = _team_season_totals(df)

    d = _dedupe_player_seasons(df)
    d = d.merge(team_totals, on=['Tm', 'Year'], how='left')
    d = d.sort_values(['_player_key', 'Year'])

    player_key, year = d['_player_key'], d['Year']

    ppr0 = pd.to_numeric(d['PPR'], errors='coerce').fillna(0)
    g0 = pd.to_numeric(d['G'], errors='coerce').fillna(0)
    touches0 = pd.to_numeric(d['Att.1'], errors='coerce').fillna(0) + pd.to_numeric(d['Rec'], errors='coerce').fillna(0)
    tgt_share0 = pd.to_numeric(d['Tgt'], errors='coerce') / d['Team_Tgt'].replace(0, np.nan)
    rush_share0 = pd.to_numeric(d['Att.1'], errors='coerce') / d['Team_Rush_Att'].replace(0, np.nan)

    def lag(values, n):
        return _lag_with_year_gate(player_key, year, values, n)

    ppr1, ppr2 = lag(ppr0, 1), lag(ppr0, 2)
    touches1, touches2 = lag(touches0, 1), lag(touches0, 2)
    g1 = lag(g0, 1)
    tgt_share1, tgt_share2 = lag(tgt_share0, 1), lag(tgt_share0, 2)
    rush_share1, rush_share2 = lag(rush_share0, 1), lag(rush_share0, 2)

    out = pd.DataFrame(index=d.index)
    out['_player_key'] = player_key
    out['Year'] = year

    out['PPR_Weighted_3yr'] = _weighted_avg_3yr(ppr0, ppr1, ppr2)
    out['PPR_Trend_Slope'] = _trend_slope_3yr(ppr0, ppr1, ppr2)
    out['Touches_Weighted_3yr'] = _weighted_avg_3yr(touches0, touches1, touches2)
    out['Touches_Trend_Slope'] = _trend_slope_3yr(touches0, touches1, touches2)

    # Recency-focused volatility: std across whichever of the last up to 3
    # seasons are available, vs. PPR_Std's whole-career std - more sensitive
    # to a player's CURRENT boom/bust phase than a decade-old rough patch
    # (and, being built from this same all_data lookup, doesn't inherit
    # PPR_Std's collapse-to-0 problem at prediction time).
    recent_window = pd.concat([ppr0.rename(0), ppr1.rename(1), ppr2.rename(2)], axis=1)
    out['Recent_PPR_Std'] = recent_window.std(axis=1, ddof=1).fillna(0)

    # Durability history over the last 2 seasons (this one plus the prior),
    # instead of just the single prior-season gap Games_Missed_Prev already
    # captures.
    season_len = 17
    missed0 = (season_len - g0).clip(lower=0)
    missed1 = (season_len - g1).clip(lower=0)
    out['Games_Missed_2yr'] = missed0 + missed1.fillna(0)

    out['Tgt_Share'] = tgt_share0
    out['Tgt_Share_Trend'] = _trend_slope_3yr(tgt_share0.fillna(0), tgt_share1, tgt_share2)
    out['Rush_Share'] = rush_share0
    out['Rush_Share_Trend'] = _trend_slope_3yr(rush_share0.fillna(0), rush_share1, rush_share2)

    return out


_MULTI_YEAR_FEATURES = None


def _get_multi_year_features():
    """Lazily build-and-cache _build_multi_year_features() - it's the same
    table regardless of position or backtest fold, so compute it once per
    process instead of once per add_contextual_features call."""
    global _MULTI_YEAR_FEATURES
    if _MULTI_YEAR_FEATURES is None:
        _MULTI_YEAR_FEATURES = _build_multi_year_features()
    return _MULTI_YEAR_FEATURES


def add_contextual_features(data):
    """Add contextual features that explain major prediction errors"""

    # 1. TEAM CHANGE PENALTY
    data_sorted = data.sort_values(['Player', 'Age'])
    data_sorted['Team_Change'] = data_sorted.groupby('Player')['Tm'].transform(lambda x: (x != x.shift(1)).astype(int))
    data_sorted['Team_Change'] = data_sorted['Team_Change'].fillna(0)

    # 2. INJURY RECOVERY INDICATOR
    # Players under 28 will see boost after injury while older players will be penalized
    prev_games = pd.to_numeric(data_sorted.groupby('Player')['G'].shift(1), errors='coerce')
    data_sorted['Games_Missed_Prev'] = 17 - prev_games
    data_sorted['Games_Missed_Prev'] = data_sorted['Games_Missed_Prev'].fillna(0)

    # Injury recovery boost for young players
    data_sorted['Injury_Recovery'] = np.where(
        (data_sorted['Games_Missed_Prev'] > 8) & (data_sorted['Age'] < 28), 1,
        np.where((data_sorted['Games_Missed_Prev'] > 8) & (data_sorted['Age'] >= 28), 0.7, 0)
    )

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

    # 8. EXPERIENCE/ROOKIE FEATURES
    data_sorted['Experience'] = data_sorted.groupby('Player').cumcount() + 1
    data_sorted['Is_Rookie'] = np.where(data_sorted['Experience'] == 1, 1, 0)
    data_sorted['Is_Sophomore'] = np.where(data_sorted['Experience'] == 2, 1, 0)

    # 9. MULTI-YEAR TREND / OPPORTUNITY-SHARE / CONSISTENCY FEATURES
    # Looked up by (player, feature-season) from the full 2000-2025 history
    # in all_data - see _build_multi_year_features's docstring for why this
    # is looked up rather than computed the same way (7)/(8) above are: this
    # function also runs on a single-season snapshot at prediction/backtest
    # time, where a groupby('Player') shift computed only from `data` would
    # have no history left to look back on.
    #
    # Note: build_prediction_features runs add_team_context_features (a
    # separate module) before this one; when team_data/{year}/ exists that
    # merges in a team-context table with its own 'Year' column, which
    # pandas silently renames this frame's 'Year' to 'Year_x' rather than
    # erroring since the merge key is Tm/Team, not Year. Recover whichever
    # one is actually this row's own feature-season year.
    year_col = 'Year' if 'Year' in data_sorted.columns else 'Year_x'
    data_sorted['_player_key'] = _clean_player_key(data_sorted['Player'])
    data_sorted['_feature_year'] = pd.to_numeric(data_sorted[year_col], errors='coerce')
    history = _get_multi_year_features().rename(columns={'Year': '_feature_year'})
    data_sorted = data_sorted.merge(
        history, on=['_player_key', '_feature_year'], how='left', validate='m:1'
    )
    data_sorted = data_sorted.drop(columns=['_player_key', '_feature_year'])

    return data_sorted

def add_advanced_position_features(data, position):
    """Add position-specific advanced features - combines enhanced_features.py (priority) + predict_position_specific.py features"""

    if position == 'QB':
        # Enhanced features (priority)
        data['Pass_Att_PG'] = pd.to_numeric(data['Att'], errors='coerce') / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)
        data['QB_Efficiency'] = pd.to_numeric(data['TD'], errors='coerce') / pd.to_numeric(data['Att'], errors='coerce').replace(0, 1) * 100
        data['Turnover_Rate'] = (pd.to_numeric(data['Int'], errors='coerce') + pd.to_numeric(data['FL'], errors='coerce')) / pd.to_numeric(data['Att'], errors='coerce').replace(0, 1) * 100
        prev_att = pd.to_numeric(data.groupby('Player')['Att'].shift(1), errors='coerce')
        data['Low_Attempts_Prev'] = np.where(prev_att < 200, 1, 0)

        # Additional features from predict_position_specific.py
        data['Rush_Att'] = pd.to_numeric(data['Att.1'], errors='coerce').fillna(0)
        data['Rush_Yds'] = pd.to_numeric(data['Yds.1'], errors='coerce').fillna(0)
        data['Rush_TD'] = pd.to_numeric(data['TD.1'], errors='coerce').fillna(0)
        data['Rush_YPG'] = data['Rush_Yds'] / data['G'].replace(0, 1)
        data['Rush_Att_PG'] = data['Rush_Att'] / data['G'].replace(0, 1)
        data['QB_Mobility_Score'] = data['Rush_YPG'] + (data['Rush_Att_PG'] * 2)

        # QBs decline much later than RB/WR/TE - often still improving into
        # their 30s on experience/reads - so give the position its own,
        # later-onset penalty curve instead of sharing RB's.
        data['QB_Age_Penalty'] = np.where(data['Age'] > 37, (data['Age'] - 37) ** 1.5, 0)

    elif position == 'RB':
        # Enhanced features (priority)
        data['Snap_Share_Proxy'] = pd.to_numeric(data['G'], errors='coerce') / 17
        rush_tds = pd.to_numeric(data['TD.1'], errors='coerce').fillna(0)
        data['Goal_Line_Upside'] = rush_tds / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)
        data['Workload_Bonus'] = np.where(data['Workload_Premium'] == 1, 15, 0)
        data['Elite_Workload_Bonus'] = np.where(data['Elite_Workload'] == 1, 25, 0)
        data['Proven_Starter_Bonus'] = np.where(data['Proven_Starter'] == 1, 20, 0)
        data['Young_Opportunity'] = np.where(
            (pd.to_numeric(data['Age'], errors='coerce') < 26) & (data['Committee_Risk'] == 0), 1, 0
        )

        # Additional features from predict_position_specific.py
        data['Touches_PG'] = data['Total_Touches'] / data['G'].replace(0, 1)
        data['Target_Share'] = pd.to_numeric(data['Tgt'], errors='coerce').fillna(0) / data['G'].replace(0, 1)
        data['RB_Age_Penalty'] = np.where(data['Age'] > 27, (data['Age'] - 27) ** 1.5, 0)
        data['High_Workload'] = np.where(data['Touches_PG'] > 18, 1, 0)

        # Sharper, later-forming complement to the smoother RB_Age_Penalty
        # curve above: an explicit indicator for the well-documented "RB
        # cliff" rather than relying on the tree to rediscover the threshold.
        data['RB_Age_Cliff'] = np.where(data['Age'] >= 29, 1, 0)

    elif position in ['WR', 'TE']:
        # Enhanced features
        rec_yds = pd.to_numeric(data['Yds.2'], errors='coerce').fillna(0)
        data['Target_Efficiency'] = rec_yds / pd.to_numeric(data['Tgt'], errors='coerce').replace(0, 1)
        rec_tds = pd.to_numeric(data['TD.2'], errors='coerce').fillna(0)
        data['Red_Zone_Value'] = rec_tds / pd.to_numeric(data['G'], errors='coerce').replace(0, 1)
        data['Target_Regression_Protection'] = np.where(data['Elite_Target_Share'] == 1, 25, 0)
        data['Proven_Target_Bonus'] = np.where(
            (pd.to_numeric(data.groupby('Player')['Tgt'].shift(1), errors='coerce') > 120) &
            (data['Elite_Target_Share'] == 0), 10, 0
        )

        data['Target_PG'] = pd.to_numeric(data['Tgt'], errors='coerce').fillna(0) / data['G'].replace(0, 1)
        data['Catch_Rate'] = pd.to_numeric(data['Rec'], errors='coerce').fillna(0) / pd.to_numeric(data['Tgt'], errors='coerce').replace(0, 1)
        data['YPG'] = pd.to_numeric(data['Yds.2'], errors='coerce').fillna(0) / data['G'].replace(0, 1)

        # WR/TE decline later than RB - no age-curve feature existed for
        # this group before; onset threshold set later than RB's.
        data['WR_TE_Age_Penalty'] = np.where(data['Age'] > 30, (data['Age'] - 30) ** 1.5, 0)

    return data

# def build_enhanced_model(position):
#     """Build model with enhanced contextual features"""
#     import sys
#     sys.path.append('..')
#     from data_cleaning.create_df import all_data

#     # Load and enhance data
#     data = all_data.copy()
#     data['PPR_Points'] = pd.to_numeric(data['PPR'], errors='coerce')
#     data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(25)
#     data['G'] = pd.to_numeric(data['G'], errors='coerce').fillna(0)

#     # Convert key numeric columns
#     numeric_cols = ['Att', 'TD', 'Int', 'Tgt', 'Rec', 'Yds', 'FL']
#     for col in numeric_cols:
#         if col in data.columns:
#             data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

#     # Add contextual features
#     data = add_contextual_features(data)

#     # Filter by position and add position-specific features
#     pos_data = data[data['FantPos'] == position].copy()
#     pos_data = add_advanced_position_features(pos_data, position)

#     # Enhanced feature set
#     base_features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Tgt', 'Rec', 'Y/R', 'Y/A', 'Fmb', 'FL']

#     contextual_features = [
#         'Team_Change', 'Games_Missed_Prev', 'Injury_Recovery',
#         'Low_Usage_High_Potential', 'High_Prev_Performance',
#         'PPR_Std', 'Seasons_Played'
#     ]

#     # Position-specific features
#     if position == 'QB':
#         pos_features = ['Pass_Att_PG', 'QB_Efficiency', 'Turnover_Rate', 'Low_Attempts_Prev', 'QB_Streaming_Penalty']
#     elif position == 'RB':
#         pos_features = ['Snap_Share_Proxy', 'Goal_Line_Upside', 'Young_Opportunity', 'Workload_Bonus', 'Elite_Workload_Bonus', 'Proven_Starter_Bonus', 'Committee_Risk']
#     else:
#         pos_features = ['Target_Efficiency', 'Red_Zone_Value', 'Target_Regression_Protection', 'Proven_Target_Bonus']

#     all_features = base_features + contextual_features + pos_features

#     # Prepare feature matrix
#     feature_cols = [col for col in all_features if col in pos_data.columns]
#     features_df = pos_data[feature_cols].copy()

#     for col in features_df.columns:
#         features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

#     print(f"{position}: Using {len(feature_cols)} features including contextual factors")
#     print(f"Key new features: {contextual_features[:3]}")

#     return features_df, pos_data['PPR_Points']

# if __name__ == "__main__":
#     # Test the enhanced features
#     for pos in ['QB', 'RB', 'WR', 'TE']:
#         try:
#             X, y = build_enhanced_model(pos)
#             print(f"{pos}: {len(X)} samples, {X.shape[1]} features")
#         except Exception as e:
#             print(f"{pos}: Error - {e}")
