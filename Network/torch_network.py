from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import torch
#import torch.nn as nn

CSV_PATH = Path("Data/COVID-19.csv")


def get_data(path):
    country_data = []

    data_path = Path.cwd() / path
    data_frame = pd.read_csv(data_path, usecols=[
        "cases", "deaths", "countriesAndTerritories", "dateRep"])
    group = data_frame.groupby("countriesAndTerritories")

    for country, df_country in group:
        cases = df_country["cases"]
        last_index = cases.to_numpy().nonzero()[0][-1] + 1
        df_trimmed = df_country.drop(df_country.index[last_index:])
        df_reversed = df_trimmed.iloc[::-1]
        country_data.append((country, df_reversed))
        
    return country_data


def plot_data(country_data):
    for country, df_country in country_data:
        if country == "Germany":
            df_country.plot(x="dateRep")
            plt.grid()
            plt.title("Country: " + country)
            plt.ylabel('Number of people')
            plt.xlabel('Date')
            plt.show()


def main():
    data = get_data(CSV_PATH)
    #plot_data(data)


if __name__ == "__main__":
    main()
