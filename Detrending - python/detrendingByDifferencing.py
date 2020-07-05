from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

import sys
from math import log10
import math as m


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



def detrendWeekdays(data):
    averageDay = 0
    model = [1 for i in range(7)]
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



def removeTrendFromData(x,y):
    for i in range(len(x)):
        x[i] = x[i] - y[i]
    return x

def detrendByDifferencing(data):
    newData = [0 for i in range(len(data)-1)]
    for i in range(len(data)-1):
        newData[i] = data[i+1] - data[i]
    return newData


def main():


    data = get_data(CSV_PATH)


    f = open("detrendedDataAllCountries", "w")

    f.write("XXX 2\n")

    i = 0

    for c, df_country in data:
        deaths, cases = unpackData(data, c)

        if (len(cases) < 31):
            continue
    

        if (detrendOneCountry(deaths, cases, c, f)):
            i+=1

    print(i)
    
    f.close()

def confirm():
    x = input()
    if (x != ''):
        return False
    return True

def plotData(data1, description1, data2, description2):
    plt.plot(data1, label = description1)
    plt.plot(data2, label = description2)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def detrendOneCountry(deaths, cases, c, f):

    plotData(deaths, "originalDeaths", cases, "originalCases")


    if (not confirm()):
        print("data not used")
        return False


    #removes weekly oscillation of data. The model contains the average proportion of cases at a specific weekday compard to the average. 
    modelWeekdaysCases, cases = detrendWeekdays(cases)



    #creates a trend using a polynominal function of the given order. The model contain the polynomial coefficients, highest power first.
    deaths = detrendByDifferencing(deaths)
    cases = detrendByDifferencing(cases)


    #Normalize data
    for i in range(len(cases)):

        if (deaths[i] > 1):
            deaths[i] = m.pow(deaths[i],float(1)/3)
        else:
            deaths[i] = m.pow(abs(deaths[i]),float(1)/3) * -1

        if (cases[i] > 1):
            cases[i] = m.pow(cases[i],float(1)/3)
        else:
            cases[i] = m.pow(abs(cases[i]),float(1)/3) * -1


    f.write(c + " " + str(len(cases)) + "\n")

    for i in range(len(cases)):
        f.write(str(deaths[i]) + " " + str(cases[i])+ "\n")

    print("Data is used")
    return True


if __name__ == "__main__":
    main()
