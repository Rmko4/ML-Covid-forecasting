from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np


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


def unpackData(data, country):
    deaths = []
    cases  = []
    for c, df_country in data:
        if c == country:
            for i in range(len(df_country.cases.values)):
                deaths.append(df_country.deaths.values[i])
                cases.append(df_country.cases.values[i])

    return deaths, cases


def trendWeekdays(data):

    model = [0 for i in range(7)]
    numberDatapoints = [0 for i in range(7)]

    for i in range(len(data)):
        model[(i % 7)] += data[i]
        numberDatapoints[(i % 7)] +=1
    

    for i in range(7):
        model[i] = model[i] / numberDatapoints[i]

    trend =[]
    for i in range(len(data)):
        trend.append(data[i] - model[(i % 7)])

    return model, trend


def removeTrendFromData(data, trend):
    newData = []
    for i in range(len(data)):
        newData.append(data[i] - trend[i])
    return newData

def detrendByDifferences(data):
    differenceData = []
    for i in range(len(data) -1):
        differenceData.append(data[i] - data[i+1])
    return differenceData

def main():
    data = get_data(CSV_PATH)

    #make deaths and cases simple arrays
    deaths, cases = unpackData(data, "Germany")

    #show original data
    plt.plot(deaths)
    plt.plot(cases)
    plt.show()



    deathsDifferences = detrendByDifferences(deaths)
    casesDifferences = detrendByDifferences(cases)

    #show differences without weekly detrending
    plt.plot(deathsDifferences)
    plt.plot(casesDifferences)
    plt.show()


    #removes weekly oscillation of data. The model contains the average difference each weekday. 
    modelWeekdaysDeaths,  trendWeekDeaths= trendWeekdays(deathsDifferences)
    modelWeekdaysCases, trendWeekCases = trendWeekdays(deathsDifferences)


    detrendedDeaths = removeTrendFromData(deathsDifferences, trendWeekDeaths)
    detrendedCases = removeTrendFromData(casesDifferences, trendWeekCases)


    #show detrended data
    plt.plot(detrendedCases, label = "random noise left after detrending cases" )
    plt.plot(detrendedDeaths, label = "random noise left after detrending deaths" )
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    main()
