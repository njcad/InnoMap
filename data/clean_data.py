"""
Script to clean dblp-v10.csv data.
Produces clean csv file containing only entries from dblp-v10.csv that have valid values for
    1. Title
    2. Abstract
    3. References
    4. ID
"""

import pandas as pd 

# read in un-clean dataset
dblp_path = "dblp-v10.csv"
df = pd.read_csv(dblp_path)

# filter on these categories
categories = ["abstract", "references", "title", "id"]
clean_df = df.dropna(subset=categories)

# new dataset
data_clean_path = "data_clean.csv"
clean_df.to_csv(data_clean_path, index=False)
