import pandas as pd
import glob
import os
import re

# Get the directory of this script and build paths relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
stats_dir = os.path.join(project_root, 'stats')

def load_fantasy_stats(path):
    """Load a Pro-Football-Reference fantasy-stats export.

    PFR's export has shipped two formats across the years in stats/: a single
    header row, and a double (grouped) header row (e.g. "Passing" spanning
    several sub-columns). Reading every file with the same fixed header
    offset silently mis-parses whichever format doesn't match - real columns
    like 'FantPos' become 'Unnamed: 0_level_0' and the true header row is
    swallowed as if it were a data row - so every score/feature drawn from
    that file downstream is either missing or wrong. Detect the format per
    file instead of assuming one.

    Also tags each row with the season Year parsed from the filename, since
    nothing upstream otherwise records which season a row belongs to.
    """
    df = pd.read_csv(path)
    if 'FantPos' not in df.columns:
        df = pd.read_csv(path, header=1)

    year_match = re.search(r'(\d{4})', os.path.basename(path))
    df['Year'] = int(year_match.group(1)) if year_match else None
    return df

# Use only fantasy stats to reduce memory usage (using all the data was crashing)
fantasy_files = sorted(glob.glob(os.path.join(stats_dir, 'fantasy_stats_for_*.csv')))
fantasy_dfs = [load_fantasy_stats(file) for file in fantasy_files]

# Combine all fantasy data
all_data = pd.concat(fantasy_dfs, ignore_index=True)
all_data = all_data.fillna(0)

print(f"Loaded {len(all_data)} records from {len(fantasy_files)} files")

all_data.to_pickle('../stats/all_data_pickle.pkl')
