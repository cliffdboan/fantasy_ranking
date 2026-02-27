import pandas as pd
from time import sleep
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# This failed after 2025, likely due to a PFR site change against bots.
# Since I already had data from 2000-2024, I just downloaded a CSV from the site and put it in the stats folder.

def scrape_x_stats(type: str, years: list) -> None:
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
            print(f"Saved {type}_stats_for_{year}.csv")

            # PFR will block you after a certain number of attempts, and let you go after an hour
            # Uncomment below to skirt around a little better by waiting a minute between calls
            # sleep(60)
        except Exception as e:
            print(f"Could not scrape {type} stats for {year}. {e}")


# Used to scrape multiple years of data. If you want to scrape just one year, call
# the function directly with the year as a string, e.g. scrape_x_stats(stat_types, ['2025'])
years = []
for i in range (2020, 2026):
    years.append(str(i))
stat_types = ['fantasy'] # Found that we only use fantasy stats for the model

for type in stat_types:
    scrape_x_stats(type, [2025])
