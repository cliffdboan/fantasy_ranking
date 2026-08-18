# Fantasy Football Prediction Pipeline

Predicts PPR fantasy football points for the upcoming NFL season, position by position (QB/RB/WR/TE), using 26 years of historical stats plus team-level context (opponent strength, offensive line, projected volume). Built to be re-run every year: collect the new season's context, run one command, get ranked predictions.

## How it works

Each position gets its own model, trained to forecast a player's **next** season from their **most recent completed** season - the model never sees a season's own box score as a feature for predicting that same season's points (that would just be reconstructing the scoring formula, not forecasting anything). Training data is every player-season back to 2000, reshaped into `(season S stats -> season S+1 PPR)` pairs.

- **QB / RB**: `RandomForestRegressor` (scikit-learn)
- **WR / TE**: `ExtraTreesRegressor` (scikit-learn)

Features come from three places: the player's own recent box score and trend (multi-year weighted averages, trend slope, opportunity share, age curves), team context (opponent defense rank vs. the position, offensive line rank, projected pass/rush volume - only really populated for the two most recent seasons, see [Limitations](#known-limitations)), and a handful of manual post-hoc adjustments for known patterns (e.g. backup QBs stepping into a starting role).

Model choice, feature list, and hyperparameters are all validated with a walk-forward backtest (`build_model/backtest.py`) - train on strictly-past data, predict a held-out year, score against what actually happened - rather than a single spot-checked season. As of the last full validation (2010-2025, 64 position-year folds): **avg MAE 46.3, Pearson 0.704, Spearman 0.684, top-20 hit rate 0.625**. Re-run the backtest after any modeling change; don't trust a change that only looks better on one year.

The final adjusted-predictions file is trimmed to a draftable pool per position (`DRAFT_POOL_SIZE` in `predicting/prediction_adjustments.py` - defaults to QB 36 / RB 70 / WR 70 / TE 30, tune per your league format), so the output isn't cluttered with every player who logged any fantasy-position stats last season. The cutoff is applied to *adjusted* points, after the manual adjustments run - not to last season's raw games/points, since that would risk cutting a legitimate bounce-back candidate (e.g. a stud RB who tore an ACL in week 2 last year) before the model or the Injury_Recovery/Backup_QB_Breakout adjustments even got a say. The raw predictions file (`fantasy_predictions_position_specific_{year}.csv`) stays unfiltered, since `check_ppr_predictions.py` and `backtest.py` score against it.

## Project structure

```
main.py                          Orchestrates the full pipeline for one year: train -> predict -> adjust -> evaluate

data_cleaning/
  create_df.py                   Loads stats/fantasy_stats_for_*.csv into one DataFrame, auto-detecting
                                  PFR's two export formats and tagging each row with its season Year

build_model/
  model_core.py                  Shared core: lagged dataset construction, feature list, model selection/
                                  fitting - used by training, prediction, AND backtesting so all three can
                                  never silently drift apart from each other
  enhanced_features.py           Engineered features: contextual (injury recovery, workload, team change),
                                  position-specific, and multi-year trend/opportunity-share/age-curve
  team_context_integration.py    Merges opponent/team context into player rows
  scikit_position_model.py       CLI: trains and saves the 4 position models for a given year
  backtest.py                    CLI: walk-forward validation across many past years - the yardstick for
                                  whether any modeling change actually helps

predicting/
  predict_position_specific.py   CLI: generates raw predictions for a given year
  prediction_adjustments.py      CLI: applies manual heuristic adjustments on top of raw predictions
  check_ppr_predictions.py       CLI: scores one year's predictions against actual results, with plots

scraper/
  scraper.py                     Scrapes historical PFR season stats (see Limitations - may need a manual
                                  fallback)
  def_vs_position_scraper.py     CLI: scrapes defense-vs-position rankings for a given year into team_data/

team_data/
  populate_team_attempts.py      CLI: merges pass/rush attempt projections into a year's team context
  populate_vs_pos_context.py     CLI: merges defense-vs-position rankings into a year's team context
  calculate_team_rz_rates.py     CLI: recomputes red-zone pass rate from the prior year's red-zone data
  clean_redzone_data.py          Utility: strips extra columns from a freshly downloaded red-zone CSV
  {year}/                        Per-year team context inputs (team_context_{year}.csv, defense rankings,
                                  attempt projections, red-zone data)

stats/
  fantasy_stats_for_{year}.csv   Actual player stats per season, 2000-present - training data, and the
                                  "most recent season" input for next year's prediction

models/{year}/                  Trained model files (model_qb.pkl, model_rb.pkl, model_wr.pkl, model_te.pkl)
predictions/{year}/             Raw and adjusted prediction CSVs
model_analysis/{year}/          Feature-importance and validation plots
model_analysis/backtest/        Latest walk-forward backtest results (backtest_results.csv)

draft_builder/                  Optional: compares model predictions against expert consensus rankings and
                                 builds an HTML draft-day cheat sheet
run_ffanalytics.R               Optional: pulls expert consensus projections via R's ffanalytics package,
                                 feeding draft_builder's comparison
```

## Setup

Requires **Python 3.12** with the packages in `requirements.txt` (including Selenium, for the defense-rankings scraper, which also needs Chrome installed):

```
pip install -r requirements.txt
```

If your system's default `python3` doesn't resolve to an interpreter with these packages installed, use the specific interpreter that does (e.g. `python3.12`) for every command below.

## Running it each year

Example: transitioning from a completed 2025 season to predicting 2026.

**1. Get the completed season's actual results into `stats/`**

Download or scrape final fantasy stats for the season that just ended into `stats/fantasy_stats_for_2025.csv`. This feeds both training data and the box-score input used to predict 2026. `scraper/scraper.py` automates this against Pro-Football-Reference, but PFR changed their site in a way that broke it after 2025 - a manual CSV download from the site may be needed instead.

**2. Populate `team_data/2026/` for the upcoming season**

- Seed the base file: copy `team_data/2025/team_context_2025.csv` to `team_data/2026/team_context_2026.csv`, bump `Year` to 2026, and hand-update `HC_Change`/`OC_Change` (coaching moves) and `OL_Rank` - no script sources these automatically, they're manual research.
- Red-zone rates (optional, carries forward if skipped):
  ```
  cd team_data
  python3.12 clean_redzone_data.py 2026/redzone_pass_2025.csv
  python3.12 clean_redzone_data.py 2026/redzone_rush_2025.csv
  python3.12 calculate_team_rz_rates.py 2026
  ```
- Pass/rush attempt projections (after placing `team_pass_attempts_2026.csv` / `team_rush_attempts_2026.csv`, from your external projection source, in `team_data/2026/`):
  ```
  cd team_data
  python3.12 populate_team_attempts.py 2026
  ```
- Defense-vs-position rankings:
  ```
  cd scraper
  python3.12 def_vs_position_scraper.py 2026
  cd ../team_data
  python3.12 populate_vs_pos_context.py 2026
  ```

**3. Run the pipeline**

```
python3.12 main.py 2026
```

Trains all four position models on every season available, generates and adjusts 2026 predictions, and evaluates against actual results if `stats/fantasy_stats_for_2026.csv` already exists (it won't, yet - that step runs automatically once it does). Outputs land in `predictions/2026/`, `models/2026/`, `model_analysis/2026/`.

## Validating a model change

Before trusting any change to features, models, or hyperparameters, check it against the walk-forward backtest rather than a single year:

```
cd build_model
python3.12 backtest.py 2010 2025
```

Trains on strictly-past data for each test year in the range and scores against real results - takes a couple of minutes for the full range. Use a shorter range (e.g. `2020 2025`) for faster iteration while exploring.

## Known limitations

- **Team context has thin historical coverage.** Real `team_data/` only exists for 2024-2025; earlier training rows get neutral, imputed defaults (with a `Has_Team_Context` flag so the model can tell the difference). This gets more valuable as more `team_data/{year}/` folders accumulate over time.
- **`OL_Rank` and `HC_Change`/`OC_Change` are manual entries** - no script in this repo sources them automatically.
- **`scraper/scraper.py` (historical stats) may need a manual fallback** - see step 1 above.
- **`draft_builder/` and `predictions/add_rookies.py`** reference specific years in a couple of places (e.g. `2025`) and would need small edits to target a different year.
- **Training retrains from scratch every run** (a few minutes) rather than incrementally.
