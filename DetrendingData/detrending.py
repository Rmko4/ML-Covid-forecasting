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


def plot_data(country_data):
    for country, df_country in country_data:
        if country == "Germany":
            df_country.plot(x="dateRep")
            plt.grid()
            plt.title("Country: " + country)
            plt.ylabel('Number of people')
            plt.xlabel('Date')
            plt.show()


def unpackData(data, country):
    deaths = []
    cases  = []
    for c, df_country in data:
        if c == country:
            for i in range(len(df_country.cases.values)):
                deaths.append(df_country.deaths.values[i])
                cases.append(df_country.cases.values[i])

    return deaths, cases

def detrendWithPolynominal(y, order):

    #the x points are just the days
    x =[i for i in range(len(y))]

    model = np.polyfit(x , y, order)
    predicted = np.polyval(model, x)
    return model, predicted

def detrendWeekdays(data):
    averageDay = 0
    model = [0 for i in range(7)]

    numberDatapoints = [0 for i in range(7)]
    for i in range(len(data)):
        averageDay += data[i]
        model[(i % 7)] += data[i]
        numberDatapoints[(i % 7)] +=1
    

    averageDay /= len(data)

    for i in range(7):
        model[i] = model[i] / numberDatapoints[i]
        model[i] = model[i] / averageDay


    predictedData = []
    for i in range(len(data)):
        predictedData.append(data[i] * model[(i % 7)])

    return model, predictedData

def removeTrendFromData(data, trendData):
    result = []
    for i in range(len(data)):
        result.append(data[i] - trendData[i])
    return result

def  predictWithModel(modelPolynominals, modelWeekdays, day):

     return np.polyval(modelPolynominals, day) * modelWeekdays[(day % 7)]


def main():
    data = get_data(CSV_PATH)

    #make deaths and cases simple arrays
    deaths, cases = unpackData(data, "Germany")

    #creates a trend using a polynominal function of the given order. The model contain the polynomial coefficients, highest power first.
    modelPolynominalsDeaths, trendDeaths = detrendWithPolynominal(deaths, 15)
    modelPolynominalsCases,  trendCases = detrendWithPolynominal(cases, 15)

    #removes weekly oscillation of data. The model contains the average proportion of cases at a specific weekday compard to the average. 
    modelWeekdaysDeaths, trendDeaths = detrendWeekdays(trendDeaths)
    modelWeekdaysCases, trendCases = detrendWeekdays(trendCases)

    plt.plot(deaths)
    plt.plot(cases)
    plt.plot(trendDeaths)
    plt.plot(trendCases)
    plt.show()

    #remove trend from data
    deaths = removeTrendFromData(deaths, trendDeaths)
    cases = removeTrendFromData(cases, trendCases)

    plt.plot(deaths)
    plt.plot(cases)
    plt.show()

    #use the two models to predict the next day
    predictionNextDayDeaths = predictWithModel(modelPolynominalsDeaths, modelWeekdaysDeaths, len(deaths) + 1)
    predictionNextDayCases = predictWithModel(modelPolynominalsCases, modelWeekdaysCases, len(deaths) + 1)

    print("The trend-model is predicting {0} corona deaths for the next day".format(predictionNextDayDeaths))
    print("The trend-model is predicting {0} corona cases for the next day".format(predictionNextDayCases))



    #plot_data(data)


if __name__ == "__main__":
    main()
