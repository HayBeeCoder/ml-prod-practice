import pandas as pd

def load_collections(file_path="rent_apartments.csv"):
    """
    Load collections from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file containing collections data.

    Returns:
    pd.DataFrame: A DataFrame containing the collections data.
    """
    df = pd.read_csv(file_path)
    return df
  
    