import pandas as pd

def remove_vacancy_columns(year=2025):
    """Remove target and carry vacancy columns from team context"""

    # Load team context
    team_context = pd.read_csv(f'{year}/team_context_{year}.csv')

    # Remove vacancy columns
    columns_to_remove = ['Target_Vacancy_WR', 'Target_Vacancy_TE', 'Target_Vacancy_RB', 'Carry_Vacancy_RB']
    team_context_clean = team_context.drop(columns=columns_to_remove)

    # Save cleaned file
    team_context_clean.to_csv(f'{year}/team_context_{year}.csv', index=False)

    print("Removed vacancy columns from team context!")
    print(f"Remaining columns: {list(team_context_clean.columns)}")
    print(f"\nSample data:")
    print(team_context_clean.head(3))

if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    remove_vacancy_columns(year)