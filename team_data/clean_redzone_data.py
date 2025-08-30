import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Please provide a redzone data file")
    print("Usage: python clean_redzone_data.py <redzone_data_file>")
    sys.exit()
redzone_data = sys.argv[1]


def clean_redzone_data(rz_file):
    """Clean redzone redzone data by removing last two columns"""

    # Load the data
    df = pd.read_csv(rz_file)

    # Remove the last two columns
    df_clean = df.iloc[:, :-2]

    # Save cleaned data
    df_clean.to_csv(rz_file, index=False)

    print("Cleaned redzone data saved!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Cleaned columns: {len(df_clean.columns)}")
    print(f"Removed columns: {list(df.columns[-2:])}")

if __name__ == "__main__":
    clean_redzone_data(redzone_data)
