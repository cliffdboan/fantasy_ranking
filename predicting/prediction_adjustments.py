import pandas as pd
import numpy as np

def apply_prediction_adjustments(predictions_df, year=2025):
    """Apply manual adjustments based on known prediction patterns"""
    
    # Create adjustment rules based on our analysis
    adjustments = []
    
    for idx, row in predictions_df.iterrows():
        player = row['Player'].replace('*', '').replace('+', '')
        position = row['Position']
        predicted = row['Predicted_Fantasy_Points']
        
        adjustment_factor = 1.0
        reason = ""
        
        # 1. INJURY RECOVERY BOOST (addresses J.K. Dobbins, De'Von Achane misses)
        # Use low last season points + young age as proxy for injury recovery
        last_season = row.get('Last_Season_Points', 0)
        age = row.get('Age', 30)
        if last_season < 50 and age < 28 and position == 'RB':
            adjustment_factor *= 1.6  # 60% boost for young RBs with low previous production
            reason += "Injury_Recovery+"
        
        # 2. BACKUP QB BREAKOUT (addresses Sam Darnold situation)
        # Use low prediction + low last season as proxy for backup QB
        if position == 'QB' and predicted < 100 and last_season < 150:
            adjustment_factor *= 2.2  # Major boost for backup QBs getting opportunity
            reason += "Backup_QB_Breakout+"
        
        # 3. HIGH VARIANCE PENALTY (addresses overvaluation issues)
        if predicted > 250:
            adjustment_factor *= 0.9  # 10% penalty for high predictions
            reason += "High_Prediction_Penalty+"
        
        # 4. YOUNG RB OPPORTUNITY (addresses Chase Brown situation)
        # Use young age + low previous production as proxy
        if position == 'RB' and age < 25 and last_season < 100:
            adjustment_factor *= 1.5  # 50% boost for young RBs with opportunity
            reason += "Young_RB_Opportunity+"
        
        # 5. OVERVALUATION PROTECTION (addresses Aiyuk, Olave situations)
        if position in ['WR', 'TE'] and predicted > 200:
            # Check if player had high production last year (regression risk)
            if last_season > 150:
                adjustment_factor *= 0.85  # 15% penalty for potential regression
                reason += "Target_Regression_Risk+"
        
        adjusted_prediction = predicted * adjustment_factor
        
        adjustments.append({
            'Player': player,
            'Position': position,
            'Original_Prediction': predicted,
            'Adjustment_Factor': adjustment_factor,
            'Adjusted_Prediction': adjusted_prediction,
            'Reason': reason.rstrip('+') if reason else 'None'
        })
    
    # Apply adjustments
    adj_df = pd.DataFrame(adjustments)
    predictions_df['Adjusted_Fantasy_Points'] = adj_df['Adjusted_Prediction']
    predictions_df['Adjustment_Reason'] = adj_df['Reason']
    
    # Re-rank based on adjusted predictions
    predictions_df = predictions_df.sort_values('Adjusted_Fantasy_Points', ascending=False)
    predictions_df['Adjusted_Rank'] = range(1, len(predictions_df) + 1)
    
    return predictions_df

def show_adjustment_summary(predictions_df):
    """Show summary of adjustments made"""
    
    # Count adjustments by type
    adjustments = predictions_df[predictions_df['Adjustment_Reason'] != 'None']
    
    print(f"\nADJUSTMENT SUMMARY")
    print(f"   • Total players adjusted: {len(adjustments)}")
    
    # Show biggest adjustments
    adjustments['Adjustment_Size'] = abs(adjustments['Adjusted_Fantasy_Points'] - adjustments['Predicted_Fantasy_Points'])
    top_adjustments = adjustments.nlargest(5, 'Adjustment_Size')
    
    print(f"\nTOP 5 ADJUSTMENTS:")
    for i, (_, row) in enumerate(top_adjustments.iterrows(), 1):
        change = row['Adjusted_Fantasy_Points'] - row['Predicted_Fantasy_Points']
        direction = "↑" if change > 0 else "↓"
        print(f"   {i}. {row['Player']} ({row['Position']}) {direction}{abs(change):.0f} pts")
        print(f"      {row['Predicted_Fantasy_Points']:.0f} → {row['Adjusted_Fantasy_Points']:.0f} | {row['Adjustment_Reason']}")

if __name__ == "__main__":
    # Test with current predictions
    try:
        predictions = pd.read_csv('../predictions/fantasy_predictions_position_specific_2025.csv')
        adjusted = apply_prediction_adjustments(predictions)
        show_adjustment_summary(adjusted)
        
        # Save adjusted predictions
        adjusted.to_csv('../predictions/fantasy_predictions_adjusted_2025.csv', index=False)
        print(f"\nAdjusted predictions saved!")
        
    except FileNotFoundError:
        print("No predictions file found. Run predict_position_specific.py first.")