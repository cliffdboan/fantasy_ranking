import pandas as pd
import glob
import os

# Get the directory of this script and build paths relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
stats_dir = os.path.join(project_root, 'stats')

# Use only fantasy stats to reduce memory usage (using all the data was crashing)
fantasy_files = glob.glob(os.path.join(stats_dir, 'fantasy_stats_for_*.csv'))
fantasy_dfs = [pd.read_csv(file) for file in fantasy_files]

# Combine all fantasy data
all_data = pd.concat(fantasy_dfs, ignore_index=True)
all_data = all_data.fillna(0)

print(f"Loaded {len(all_data)} records from {len(fantasy_files)} files")

all_data.to_pickle('../stats/all_data_pickle.pkl')
