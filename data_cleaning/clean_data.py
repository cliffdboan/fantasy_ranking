import pandas as pd
import os

def get_all_filepaths(root):
    """Returns a list of all filepaths within the given root dir
    root: A string that defines the root direectory ('.', './stats', etc.)
    """
    filepaths = []
    for rt, dirs, files in os.walk(root):
        for file in files:
            filepaths.append(os.path.join(rt, file))

    return filepaths

for file in get_all_filepaths('./stats'):
    # First we're going to remove any duplicate rows (mainly rows that repeat the header)
    df = pd.read_csv(file)
    # df.drop_duplicates(inplace=True)

    # # Also, remove any rows that begin with "Unnamed:" because many of the csv files begin with that
    # if df.columns[0].startswith("Unname"):
    #     # set columns to the first row
    #     df.columns = df.iloc[0]

    #     # drop the first row  (contains the headers)
    #     df = df.drop(df.index[0]).reset_index(drop=True)

    #     # rewrite the file to update CSV
    #     df.to_csv(file, index=False)
    #     print(f"CSV file has been updated and saved: {file}")
    # else:
    #     print(f"File {file} did not need modification.")
