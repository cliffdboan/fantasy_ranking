import pandas as pd
from datetime import datetime

def remove_vacancy_columns(year):
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

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = datetime.now().year
        print(f"No year provided, defaulting to {year} (current year). "
              f"Usage: python remove_vacancy_columns.py <year>")

    remove_vacancy_columns(year)