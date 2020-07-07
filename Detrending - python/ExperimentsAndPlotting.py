from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

import sys
from math import log10
import math as m
import copy


CSV_PATH = Path("Data/COVID-19.csv")

c = "United_States_of_America"



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


    deaths, cases = unpackData(data, c)

    
    #plt.plot(cases, label = 'new infections')
    #plt.plot(deaths, label = 'deaths') 
    #plt.legend()
    #plt.xlabel("days since outbreak")
    #plt.ylabel("n")
    #plt.title("Official corona counts in " + c)

    #plt.show()

    #input()

    if (detrendOneCountry(deaths, cases, c, f)):
        pass



def confirm():
    x = input()
    if (x != ''):
        return False
    return True    


def detrendOneCountry(deaths, cases, c, f):

    originalCases = copy.deepcopy(cases)

    #removes weekly oscillation of data. The model contains the average proportion of cases at a specific weekday compard to the average. 
    modelWeekdaysCases, cases = detrendWeekdays(cases)


    #hardcoded network predictionsCases for cases in last week
    #Germany
    #predictionsCases = [-0.75781, -1.20060, 0.60509,-1.05637,-0.55658,0.08188,-0.16332]


    #Netherlands:
    #predictionsCases = [-1.35521 ,-0.39384,0.88670 ,-0.27431 ,-0.94084 ,-0.55098,-0.19458 ]

    #Japan
    #predictionsCases = [-0.24746,0.34753,-0.18179,-0.22755,-0.25501,0.47892,0.10034]

    #USA
    #predictionsCases = [-0.88884, 0.61148, 1.91104, 0.96110, 0.45819, 0.91920, -1.02934]
    #USA correct predictions
    #predictionsCases = [17.26489, -9.39662, 19.22612, -16.86669, 11.76012, -20.71912, 22.87347]


    allZeroPredictions = [ cases[-8] for x in range(7)]


    for i in range(len(predictionsCases)):
        predictionsCases[i] = predictionsCases[i]**3

    predictionsCases[0] =  predictionsCases[0] + cases[-8]
    
    for i in range(1,7):
        predictionsCases[i] = predictionsCases[i-1] + predictionsCases[i]

    for i in range(len(predictionsCases)):
        predictionsCases[i] = predictionsCases[i] / modelWeekdaysCases[(len(originalCases) -7 + i ) %7]
        allZeroPredictions[i] = allZeroPredictions[i] / modelWeekdaysCases[(len(originalCases) -7 + i ) %7]

    predictionsCases.insert(0, originalCases[-8])
    allZeroPredictions.insert(0, originalCases[-8])


    t1 = [x for x in range(len(cases) -30, len(cases))]
    t2 = [len(cases)-8 + x for x in range(8)]

    plt.plot(t1, originalCases[-30:], label = 'real data') 
    plt.plot(t2, predictionsCases, label = 'mlp predictionsCases')
    plt.plot(t2, allZeroPredictions, label = 'all zero prediction')
    plt.legend()
    plt.xlabel("days since outbreak")
    plt.ylabel("number of new cases")
    plt.title("Mlp predictions for cases in  " + c)

    plt.show()


    input()

    #detrend by differencing
    deaths = detrendByDifferencing(deaths)
    cases = detrendByDifferencing(cases)


    #Normalize data
    for i in range(len(cases)):

        if (deaths[i] > 0):
            deaths[i] = m.pow(deaths[i],float(1)/3)
        else:
            deaths[i] = m.pow(abs(deaths[i]),float(1)/3) * -1

        if (cases[i] > 0):
            cases[i] = m.pow(cases[i],float(1)/3)
        else:
            cases[i] = m.pow(abs(cases[i]),float(1)/3) * -1

    print(cases)


    f.write(c + " " + str(len(cases)) + "\n")

    for i in range(len(cases)):
        f.write(str(deaths[i]) + " " + str(cases[i])+ "\n")

    print("Data is used")
    return True


if __name__ == "__main__":
    main()
