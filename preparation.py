import pandas as pd
import re
from collections_loader import load_collections

def prepare_data():
    return parse_garden_col(encode_cat_cols(load_collections()))

def encode_cat_cols(df):
    return pd.get_dummies(df, columns=["balcony", "parking", "furnished", "garage", "storage"], drop_first=True)

def parse_garden_col(data):
    # Set "Not present" to 0
    data.loc[data["garden"] == "Not present", "garden"] = 0
    # Extract numbers for other values
    mask = data["garden"] != 0
    data.loc[mask, "garden"] = data.loc[mask, "garden"].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
    return data
