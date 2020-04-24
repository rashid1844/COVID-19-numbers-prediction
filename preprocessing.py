import numpy as np
import scipy as sp
import random
import bokeh
import csv
import pandas as pd
import pickle
import IPython


class Preprocessing:
    def __init__(self):
        self.cases_threshold = 100  # number of cases to start considering as training input
        self.days_length = 10  # number of days to consider poly function  (Two output files)
        self.File_Name_x = 'dataset_processed/covid2_{}_{}.csv'.format(self.cases_threshold, self.days_length)
        #self.File_Name_cases_y = 'dataset_processed/covid2_{}_{}_cases_y.csv'.format(self.cases_threshold, self.days_length)
        #self.File_Name_death_y = 'dataset_processed/covid2_{}_{}_death_y.csv'.format(self.cases_threshold, self.days_length)
        self.File_Name_y = 'dataset_processed/covid2_{}_{}_y.csv'.format(self.cases_threshold, self.days_length)
        self.lockdown()
        self.get_data()
        self.preprocess()

    def lockdown(self):
        self.lockdown_list = ['Argentina', 'Australia', 'Belguim', 'China', 'Colombia', 'Czechia', 'Denmark', 'El Salvador',
                              'France', 'Germany', 'India', 'Indonesia', 'Israel', 'Italy', 'Ireland', 'Jordan', 'Kenya',
                              'Kuwait','Malaysia', 'Morocco', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Ireland',
                              'Russia', 'Saudi','Arabia', 'South', 'Africa', 'Spain', 'United', 'Kingdom']

    def get_data(self):
        self.dataset_deaths = pd.read_csv("data/time_series_covid19_deaths_global.csv").values
        self.dataset_cases = pd.read_csv("data/time_series_covid19_confirmed_global.csv").values
        self.country_dataset = pd.read_csv("country_stats.csv").values
        row_names = ['Country', 'province', 'population', 'density', 'area', 'urban',
                    'Days_since_n_cases', 'LockDown', 'longitude', 'latitude']
        cases_names = ['cases_day{}'.format(i) for i in range(1, self.days_length+1)]
        deaths_names = ['deaths_day{}'.format(i) for i in range(1, self.days_length+1)]
        row_names = row_names + cases_names + deaths_names

        with open(self.File_Name_x, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_names)

        with open(self.File_Name_y, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['new_cases', 'new_deaths'])


    def country_stat(self, country_name):
        if country_name == 'Congo (Kinshasa)':
            index = 116
        elif country_name == 'Congo (Brazzaville)':
            index = 15
        elif country_name == 'Korea, South':
            index = 27
        elif country_name == 'US':
            index = 2
        elif country_name == 'Czechia':
            index = 85
        elif country_name == 'Taiwan*':
            index = 56

        else:
            try:
                index = np.where(country_name == self.country_dataset.T[1])[0][0]
            except:
                #IPython.embed(header='test')
                return None, None, None, None

        population = self.country_dataset[index][2]/10**8  # 100 Millions
        density = self.country_dataset[index][5]/1000  # (1000 p/Km)
        area = self.country_dataset[index][6]/10**6  # ( Million Km)
        if self.country_dataset[index][10][:-1] == 'N.A':
            urban = 0.
        else:
            urban = float(self.country_dataset[index][10][:-1])/100  # to remove percentage sign (percentage out of one)

        return population, density, area, urban

    def preprocess(self):
        cases_factor = 1000
        deaths_factor = 100
        for index, row in enumerate(self.dataset_cases):
            case_index = -1
            for i in range(len(row)-4):
                if sum(row[4:4+i]) >= self.cases_threshold:
                    case_index = i
                    break
            if case_index == -1:
                continue

            population, density, area, urban = self.country_stat(row[1])
            if population is None:
                continue

            Country = row[1]
            print(Country, len(row) - 4 - case_index)
            province = row[0]
            LockDown = 1 if Country in self.lockdown_list else 0
            longitude = (row[2] + 180) / 360
            latitude = (row[3] + 180) / 360
            for day in range(case_index, len(row) - 4):  # day is the predict day
                if day <= self.days_length:
                    continue
                #self.days_length = min(day, self.days_length)
                start = day + 4 - self.days_length
                end = day + 4
                # using total values
                #y_cases = [float(sum(row[4:i]))/1000 for i in range(start, end)]
                #y_deaths = [float(sum(self.dataset_deaths[index][4:i]))/100 for i in range(start, end)]  # note: output per 100 deaths
                # using new values
                y_cases = [(row[i] - row[i-1])/cases_factor for i in range(start, end)]
                y_deaths = [(self.dataset_deaths[index][i] - self.dataset_deaths[index][i-1])/deaths_factor for i in range(start, end)]


                #current_cases = float(sum(row[4:4+day]))/1000
                #current_deaths = float(sum(self.dataset_deaths[index][4:4+day]))/100
                days_since_n_cases = (day - case_index)/100

                out_cases = (row[day+4]-row[day+3])/cases_factor  # note: output per 1000 case
                out_deaths = (self.dataset_deaths[index][day+4]-self.dataset_deaths[index][day+3])/deaths_factor  # note: output per 1000 case

                X = [Country, province,  population, density, area, urban, days_since_n_cases, LockDown, longitude, latitude]
                X = X + list(y_cases) + list(y_deaths)
                with open(self.File_Name_x, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(X)
                with open(self.File_Name_y, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([out_cases,out_deaths])




if __name__ == '__main__':
    Preprocessing()