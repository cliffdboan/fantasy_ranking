#!/usr/bin/env python3
"""
Add missing 2025 rookies to prediction model using expert projections
"""

import pandas as pd
import sys
import os

def add_rookies():
    """Add rookies from expert projections to model predictions and update draft sheet"""

    # Read current predictions and expert data
    df = pd.read_csv('2025/fantasy_predictions_adjusted_2025.csv')
    expert_df = pd.read_csv('expert_projections_clean.csv')

    # Target rookies to add (based on expert projections)
    target_rookies = [
        # TODO: Create a dynamic way to get this list
    ]

    # Filter to only rookies not already in model
    df_players = df['Player'].str.strip().values
    target_rookies = [r for r in target_rookies if r.strip() not in df_players]

    print("🔍 Searching for rookies in expert projections...")

    rookies_added = []
    for rookie_name in target_rookies:
        # Search by last name for better matching
        last_name = rookie_name.split()[-1]
        expert_match = expert_df[expert_df['Player'].str.contains(last_name, case=False, na=False)]

        if not expert_match.empty:
            expert_row = expert_match.iloc[0]

            # Create rookie entry matching model format
            rookie_entry = {
                'Player': expert_row['Player'],
                'Position': expert_row['Position'],
                'Team': expert_row.get('Team', 'TBD'),
                'Age': expert_row.get('Age', 22),  # Use expert age or default to 22
                'Last_Season_Points': 0.0,
                'Predicted_Fantasy_Points': expert_row['Expert_Points'],
                'Rank': len(df) + len(rookies_added) + 1,
                'Adjusted_Fantasy_Points': expert_row['Expert_Points'],  # Keep expert projection as-is
                'Adjustment_Reason': 'Rookie_Expert_Projection',
                'Adjusted_Rank': len(df) + len(rookies_added) + 1
            }

            rookies_added.append(rookie_entry)
            print(f"✅ Found {expert_row['Player']} ({expert_row['Position']}) - {expert_row['Expert_Points']:.1f} pts")
        else:
            print(f"⚠️  {rookie_name} not found in expert projections")

    if not rookies_added:
        print("❌ No rookies found to add")
        return

    # Convert to DataFrame and add to existing predictions
    rookie_df = pd.DataFrame(rookies_added)
    combined_df = pd.concat([df, rookie_df], ignore_index=True)

    # Re-rank by adjusted fantasy points
    combined_df = combined_df.sort_values('Adjusted_Fantasy_Points', ascending=False).reset_index(drop=True)
    combined_df['Adjusted_Rank'] = range(1, len(combined_df) + 1)

    # Save updated predictions
    combined_df.to_csv('2025/fantasy_predictions_adjusted_2025.csv', index=False)
    print(f"Added {len(rookies_added)} rookies to predictions file")

    # Update draft sheet by running comparison script
    print("Updating draft sheet...")

    # Import and run the comparison function
    sys.path.append('../draft_builder')
    from compare_predictions import compare_predictions

    # Generate new draft sheet with rookies included
    model_file = '2025/fantasy_predictions_adjusted_2025.csv'
    expert_file = 'expert_projections_clean.csv'
    output_file = '../draft_builder/draft_sheet_2025.csv'

    try:
        comparison_df = compare_predictions(model_file, expert_file, output_file)
        print("✅ Draft sheet updated successfully")

        # Show rookie rankings in final draft sheet
        rookie_names = [r['Player'] for r in rookies_added]
        rookie_rankings = comparison_df[comparison_df['Player'].isin(rookie_names)]

        if not rookie_rankings.empty:
            print("\n📊 Rookie Rankings in Draft Sheet:")
            for _, rookie in rookie_rankings.iterrows():
                expert_rank = f"Expert: {int(rookie['Expert_Rank'])}" if pd.notna(rookie['Expert_Rank']) else "No Expert Rank"
                print(f"  {rookie['Player']} ({rookie['Position']}) - Model: {int(rookie['Model_Rank'])}, {expert_rank}")

    except Exception as e:
        print(f"❌ Error updating draft sheet: {e}")
        print("Draft sheet may need to be updated manually")

if __name__ == "__main__":
    add_rookies()
