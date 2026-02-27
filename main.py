import sys
import os
import subprocess
from datetime import datetime

# This should be run after the team context is available.
# Should be done as close to your fantasy draft as possible, ideally in August
# since team DEF and predictions should be available by then.

def get_prediction_year():
    """Get prediction year from command line or user input
    Example: python main.py 2025"""
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
            if year < 2000 or year > datetime.now().year + 1:
                raise ValueError
            return year
        except ValueError:
            print(f"Invalid year: {sys.argv[1]}")
            sys.exit(1)

    while True:
        try:
            year = int(input(f"Enter prediction year (2000-{datetime.now().year + 1}): "))
            if year < 2000 or year > datetime.now().year + 1:
                raise ValueError
            return year
        except ValueError:
            print("Please enter a valid year")

def run_model_training():
    """Train the enhanced position-specific models"""
    print("\n" + "=" * 50)
    print("STEP 1: MODEL TRAINING")
    print("=" * 50)

    try:
        os.chdir('build_model')
        result = subprocess.run([sys.executable, 'scikit_position_model.py'],
                              capture_output=True, text=True)
        os.chdir('..')

        if result.returncode != 0:
            print(f"!!!!! Model training failed: {result.stderr} !!!!!")
            return False

        print(result.stdout)
        print("-----Model training completed successfully!")
        return True
    except Exception as e:
        print(f"!!!!! Model training failed: {e} !!!!!")
        return False

def run_predictions(year):
    """Generate predictions for the specified year"""
    print("\n" + "=" * 50)
    print(f"STEP 2: GENERATING {year} PREDICTIONS")
    print("=" * 50)

    try:
        os.chdir('predicting')

        # Update prediction year in the script
        with open('predict_position_specific.py', 'r') as f:
            content = f.read()

        # Replace the PREDICTION_YEAR line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('PREDICTION_YEAR = '):
                lines[i] = f'PREDICTION_YEAR = {year}'
                break

        with open('predict_position_specific.py', 'w') as f:
            f.write('\n'.join(lines))

        # Run predictions
        result = subprocess.run([sys.executable, 'predict_position_specific.py'],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print(f"!!!!! Prediction generation failed: {result.stderr} !!!!!")
            os.chdir('..')
            return False

        print(result.stdout)
        print(f"-----{year} predictions generated successfully.")
        os.chdir('..')
        return True
    except Exception as e:
        print(f"!!!!! Prediction generation failed: {e} !!!!!")
        os.chdir('..')
        return False

def run_adjustments(year):
    """Apply prediction adjustments"""
    print("\n" + "=" * 50)
    print(f"STEP 3: APPLYING ADJUSTMENTS")
    print("=" * 50)

    try:
        os.chdir('predicting')

        # Update adjustment year in the script
        with open('prediction_adjustments.py', 'r') as f:
            content = f.read()

        # Update the year in the function definition and file paths
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def apply_prediction_adjustments(predictions_df, year=' in line:
                lines[i] = f'def apply_prediction_adjustments(predictions_df, year={year}):'
            elif 'fantasy_predictions_position_specific_' in line and '.csv' in line:
                lines[i] = line.replace('2024', str(year)).replace('2025', str(year))
            elif 'fantasy_predictions_adjusted_' in line and '.csv' in line:
                lines[i] = line.replace('2024', str(year)).replace('2025', str(year))
        content = '\n'.join(lines)

        with open('prediction_adjustments.py', 'w') as f:
            f.write(content)

        # Run adjustments
        result = subprocess.run([sys.executable, 'prediction_adjustments.py'],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Adjustment application failed: {result.stderr}")
            os.chdir('..')
            return False

        print(result.stdout)
        print("✓ Adjustments applied successfully!")
        os.chdir('..')
        return True
    except Exception as e:
        print(f"✗ Adjustment application failed: {e}")
        os.chdir('..')
        return False

def run_evaluation(year):
    """Run performance evaluation if actual data is available"""
    print("\n" + "=" * 50)
    print(f"STEP 4: PERFORMANCE EVALUATION")
    print("=" * 50)

    # Check if actual data exists for evaluation
    actual_file = f'stats/fantasy_stats_for_{year}.csv'
    if not os.path.exists(actual_file):
        print(f"⚠ No actual data found for {year} - skipping evaluation")
        return True

    try:
        os.chdir('predicting')
        result = subprocess.run([sys.executable, 'check_adjusted_predictions.py'],
                              capture_output=True, text=True)
        os.chdir('..')

        if result.returncode != 0:
            print(f"✗ Performance evaluation failed: {result.stderr}")
            return False

        print(result.stdout)
        print("✓ Performance evaluation completed!")
        return True
    except Exception as e:
        print(f"✗ Performance evaluation failed: {e}")
        return False

def show_results(year):
    """Display final results"""
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)

    print(f"\n📊 {year} Fantasy Football Predictions Generated!")
    print(f"\n📁 Output Files:")
    print(f"   • Raw Predictions: predictions/fantasy_predictions_position_specific_{year}.csv")
    print(f"   • Adjusted Predictions: predictions/fantasy_predictions_adjusted_{year}.csv")
    print(f"   • Models: models/model_qb.pkl, model_rb.pkl, model_wr.pkl, model_te.pkl")

    # Show top 10 predictions
    try:
        import pandas as pd
        df = pd.read_csv(f'predictions/fantasy_predictions_adjusted_{year}.csv')
        if not os.path.exists(f'predictions/fantasy_predictions_adjusted_{year}.csv'):
            # Try the predicting directory
            df = pd.read_csv(f'predicting/../predictions/fantasy_predictions_adjusted_{year}.csv')
        print(f"\n🏆 TOP 10 PREDICTIONS FOR {year}:")
        print("-" * 50)
        top_10 = df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['Player']:20s} ({row['Position']}) - {row['Adjusted_Fantasy_Points']:6.1f} pts")
    except Exception as e:
        print(f"Could not display top predictions: {e}")

def main():
    """Main pipeline execution"""
    print("🏈 Fantasy Football Prediction Pipeline")
    print("=" * 50)

    # Get prediction year
    year = get_prediction_year()
    print(f"Generating predictions for {year}")

    # Pipeline steps
    steps = [
        ("Model Training", lambda: run_model_training()),
        ("Prediction Generation", lambda: run_predictions(year)),
        ("Adjustment Application", lambda: run_adjustments(year)),
        ("Performance Evaluation", lambda: run_evaluation(year))
    ]

    # Execute pipeline
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ Pipeline failed at: {step_name}")
            sys.exit(1)

    # Show results
    show_results(year)

if __name__ == "__main__":
    main()
