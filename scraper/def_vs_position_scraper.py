import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_defense_rankings():
    """scraper to get team defense rankings vs each position"""

    url = "https://draftedge.com/nfl/nfl-defense-vs-pos/"

    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)

        # Wait for heatmap to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.ID, "heatmap")))
        time.sleep(5)

        # Get all heatmap rect[angle]s
        rects = driver.find_elements(By.CSS_SELECTOR, ".apexcharts-heatmap-rect")
        teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
        ]

        positions = ['TE', 'WR', 'RB', 'QB']

        data = []

        for rect in rects:
            val = rect.get_attribute('val')
            j_index = rect.get_attribute('j')
            i_index = rect.get_attribute('i')

            if val and j_index and i_index:
                team_idx = int(j_index)
                pos_idx = int(i_index)

                if team_idx < len(teams) and pos_idx < len(positions):
                    data.append({
                        'team': teams[team_idx],
                        'position': positions[pos_idx],
                        'rank': int(val)
                    })

        return data

    finally:
        driver.quit()

def main():
    print("Scraping defense rankings...")

    data = scrape_defense_rankings()

    if data:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Swap rows and columns
        pivot_df = df.pivot(index='team', columns='position', values='rank')

        pivot_df.to_csv('nfl_defense_rankings.csv')

        print(f"+++++Saved {len(data)} rankings to 'nfl_defense_rankings.csv'")
        print("\nPreview:")
        print(pivot_df.head())

    else:
        print("-----No data scraped")

if __name__ == "__main__":
    main()
