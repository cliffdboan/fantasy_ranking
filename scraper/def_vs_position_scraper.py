import pandas as pd
import ssl
import os
import sys

# draftedge now server-renders a plain HTML table per position (via ?pos= query
# param) instead of the old ApexCharts heatmap, so this no longer needs Selenium.
ssl._create_default_https_context = ssl._create_unverified_context

TEAM_ABBREVIATIONS = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
}

def scrape_defense_rankings():
    """scraper to get team defense rankings vs each position"""

    positions = ['QB', 'RB', 'WR', 'TE']
    data = []

    for position in positions:
        url = f"https://draftedge.com/nfl/nfl-defense-vs-pos/?pos={position.lower()}"
        table = pd.read_html(url)[0]

        for _, row in table.iterrows():
            team_name = row['Team']
            abbreviation = TEAM_ABBREVIATIONS.get(team_name)

            if abbreviation:
                data.append({
                    'team': abbreviation,
                    'position': position,
                    'rank': int(row['Rank'])
                })
            else:
                print(f"-----Unknown team '{team_name}', skipping")

    return data

def main(year):
    print("Scraping defense rankings...")

    data = scrape_defense_rankings()

    if data:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Swap rows and columns
        pivot_df = df.pivot(index='team', columns='position', values='rank')

        os.makedirs(f'../team_data/{year}', exist_ok=True)
        output_path = f'../team_data/{year}/nfl_defense_rankings_{year}.csv'
        pivot_df.to_csv(output_path)

        print(f"+++++Saved {len(data)} rankings to '{output_path}'")
        print("\nPreview:")
        print(pivot_df.head())

    else:
        print("-----No data scraped")

if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    main(year)
