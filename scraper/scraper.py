import pandas as pd
from time import sleep
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def scrape_x_stats(type: str, years: list):
    """Creates a CSV file using the given arguments from profootballreference

    Args:
        type (string): The stats you would like to scrape. For example "fantasy" for fantasy data,
    or "passing" for passing data. See website urls for all options.

        years (list): A list of years (as strings) that you would like to scrape
    """
    for year in years:
        # direct link to an HTML table on pro-football-reference
        current_url = f"https://www.pro-football-reference.com/years/{year}/{type}.htm#{type}"
        try:
            # read the table to a csv
            df = pd.read_html(current_url)[0]
            df.to_csv(f"../stats/{type}_stats_for_{year}.csv", index=False)
        except Exception as e:
            print(f"Could not scrape {type} stats for {year}. {e}")
            # PFR will block you after a certain number of attempts, and let you go after an hour
            # Uncomment below to skirt around a little better by waiting a minute between calls
        # sleep(60)

years = []
for i in range (2023, 2024):
    years.append(str(i))
stat_types = ['passing', 'fantasy', 'receiving', 'rushing']

for type in stat_types:
    scrape_x_stats(type, years)
