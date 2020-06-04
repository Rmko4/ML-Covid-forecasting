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

def detrendWithPolynominal(y, order):

    #the x points are just the days
    x =[i for i in range(len(y))]

    model = np.polyfit(x , y, order)

    predict = np.polyval(model, x)

    return model, predict

def trendWeekdays(data):
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
        model[i] =  averageDay / model[i]

    trend =[]
    for i in range(len(data)):
        trend.append(data[i] - data[i]* model[(i % 7)])

    return model, trend

def  predictWithModel(modelPolynominals, modelWeekdays, day):

     return np.polyval(modelPolynominals, day) * modelWeekdays[(day % 7)]

def removeTrendFromData(data, trend):
    newData = []
    for i in range(len(data)):
        newData.append(data[i] - trend[i])
    return newData

def combineTrends(trend1, trend2):
    combinedTrend = []
    for i in range(len(trend1)):
        combinedTrend.append(trend1[i] + trend2[i])
    return combinedTrend


def main():
    data = get_data(CSV_PATH)

    #make deaths and cases simple arrays
    deaths, cases = unpackData(data, "Germany")


    #removes weekly oscillation of data. The model contains the average proportion of cases at a specific weekday compard to the average. 
    modelWeekdaysDeaths,  trendWeekDeaths= trendWeekdays(deaths)
    modelWeekdaysCases, trendWeekCases = trendWeekdays(cases)



    detrendedDeaths = removeTrendFromData(deaths, trendWeekDeaths)
    detrendedCases = removeTrendFromData(cases, trendWeekCases)


    #creates a trend using a polynominal function of the given order. The model contain the polynomial coefficients, highest power first.
    modelPolynominalsDeaths, trendPolyDeaths = detrendWithPolynominal(detrendedDeaths, 10)
    modelPolynominalsCases,  trendPolyCases = detrendWithPolynominal(detrendedCases, 10)

    combinedDeathTrend = combineTrends(trendPolyDeaths, trendWeekDeaths)
    combinedCasesTrend = combineTrends(trendPolyCases, trendWeekCases)

    #plot original data next to total trend
    plt.plot(deaths, label = "original Deaths")
    plt.plot(combinedDeathTrend, label = "trend Deaths")
    plt.plot(cases, label = "original Cases")
    plt.plot(combinedCasesTrend, label = "trend cases")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

    detrendedDeaths = removeTrendFromData(deaths, combinedDeathTrend)
    detrendedCases = removeTrendFromData(cases, combinedCasesTrend)

    #show detrended data
    plt.plot(detrendedCases, label = "random noise left after detrending cases" )
    plt.plot(detrendedDeaths, label = "random noise left after detrending deaths" )
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


    #use the two models to predict the next day
    predictionNextDayDeaths = predictWithModel(modelPolynominalsDeaths, modelWeekdaysDeaths, len(deaths) + 1)
    predictionNextDayCases = predictWithModel(modelPolynominalsCases, modelWeekdaysCases, len(deaths) + 1)

    print("The trend-model is predicting {0} corona deaths for the next day".format(predictionNextDayDeaths))
    print("The trend-model is predicting {0} corona cases for the next day".format(predictionNextDayCases))



    #plot_data(data)


if __name__ == "__main__":
    main()
