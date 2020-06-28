from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

import sys


CSV_PATH = Path("Data/COVID-19.csv")

polynominal = 8


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
    casesReached100 = False
    for c, df_country in data:
        if c == country:
            
            for i in range(len(df_country.cases.values)):
                if (not casesReached100):
                    if (df_country.cases.values[i] >= 100):
                        casesReached100 = True
                    else:
                        continue
                cases.append(df_country.cases.values[i])
                deaths.append(df_country.deaths.values[i])

    return deaths, cases

def detrendWithPolynominal(y, order):

    #the x points are just the days
    x = [i for i in range(len(y))]

    model = np.polyfit(x , y, order)

    predict = np.polyval(model, x)

    return model, predict

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
        model[i] =  averageDay / model[i]

    for i in range(len(data)):
        data[i] = data[i] * model[(i % 7)]

    return model, data

def  predictWithModel(modelPolynominals, modelWeekdays, day):

     return np.polyval(modelPolynominals, day) * modelWeekdays[(day % 7)]


def removeTrendFromData(x,y):
    for i in range(len(x)):
        x[i] = x[i] - y[i]
    return x



def main():


    data = get_data(CSV_PATH)


    f = open("detrendedDataAllCountries", "w")

    f.write("52 2\n")

    i =0

    for c, df_country in data:
        deaths, cases = unpackData(data, c)
        
        if ( len(cases) < 60 ):
            continue

        i+=1

        detrendOneCountry(deaths, cases, c, f)

    print(i)
    
    f.close()



def detrendOneCountry(deaths, cases, c, f):
    

    #removes weekly oscillation of data. The model contains the average proportion of cases at a specific weekday compard to the average. 
    modelWeekdaysDeaths,  deaths = detrendWeekdays(deaths)
    modelWeekdaysCases, cases = detrendWeekdays(cases)



    #creates a trend using a polynominal function of the given order. The model contain the polynomial coefficients, highest power first.
    modelPolynominalsDeaths, trendPolyDeaths = detrendWithPolynominal(deaths, polynominal)
    modelPolynominalsCases,  trendPolyCases = detrendWithPolynominal(cases, polynominal)

    predictionNextDayDeaths = predictWithModel(modelPolynominalsDeaths, modelWeekdaysDeaths, len(deaths) + 1)
    predictionNextDayCases = predictWithModel(modelPolynominalsCases, modelWeekdaysCases, len(deaths) + 1)

    #print("The trend-model is predicting {0} corona deaths for the next day".format(predictionNextDayDeaths))
    #print("The trend-model is predicting {0} corona cases for the next day".format(predictionNextDayCases))

    #plt.plot(cases, label = "detrended with weekdays cases")
    #plt.plot(deaths, label = "detrended with weekdays Deaths")
    #plt.plot(trendPolyDeaths, label = "trendPolyDeaths")
    #plt.plot(trendPolyCases, label = "trendPolyCases")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #       ncol=2, mode="expand", borderaxespad=0.)
    #plt.show()


    deaths = removeTrendFromData(deaths, trendPolyDeaths)
    cases = removeTrendFromData(cases, trendPolyCases)

    #plt.plot(deaths, label = "detrended deaths")
    #plt.plot(cases, label = "detrended cases")
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #       ncol=2, mode="expand", borderaxespad=0.)
    #plt.show()



    f.write(c + " " + str(len(cases)) + "\n")

    for i in range(len(cases)):
        f.write(str(deaths[i]) + " " + str(cases[i])+ "\n")


if __name__ == "__main__":
    main()
