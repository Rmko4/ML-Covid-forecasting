from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("Data/COVID-19.csv")

def getData(path):
    data_path = Path.cwd() / path
    print(data_path)
    df = pd.read_csv(data_path, usecols=["cases", "deaths", "countryterritoryCode"])
    return df

def main():
    data = getData(CSV_PATH)

if __name__ == "__main__":
    main()
    

    pass