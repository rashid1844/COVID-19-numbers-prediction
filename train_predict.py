import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.ensemble import AdaBoostClassifier
import time
import datetime
from tensorflow.keras.models import load_model
import IPython
import csv

class Training:
    def __init__(self):
        self.day_predict = 10  # predict for the next n days
        self.days_length = 10
        self.cases_threshold = 100
        self.import_data()
        self.train_model()
        #self.load_model()
        self.predict()

    def import_data(self):
        self.X_full = pd.read_csv('dataset_processed/covid2_100_{}.csv'.format(self.days_length)).values
        self.X = np.delete(self.X_full, (0, 1), 1)  # TO remove country and province name
        self.X = self.X.astype(np.float_)
        self.y = pd.read_csv('dataset_processed/covid2_100_{}_y.csv'.format(self.days_length)).values
        self.y = self.y.astype(np.float_)
        self.dataset_deaths = pd.read_csv("data/time_series_covid19_deaths_global.csv").values
        self.dataset_cases = pd.read_csv("data/time_series_covid19_confirmed_global.csv").values
        self.country_dataset = pd.read_csv("country_stats.csv").values
        self.lockdown_list = ['Argentina', 'Australia', 'Belguim', 'China', 'Colombia', 'Czechia', 'Denmark', 'El Salvador',
                              'France', 'Germany', 'India', 'Indonesia', 'Israel', 'Italy', 'Ireland', 'Jordan', 'Kenya',
                              'Kuwait','Malaysia', 'Morocco', 'New Zealand', 'Norway', 'Poland', 'Portugal', 'Ireland',
                              'Russia', 'Saudi','Arabia', 'South', 'Africa', 'Spain', 'United', 'Kingdom']


    def schedule(self, epoch, lr):
        if epoch<100:
            return 10**-4
        elif epoch<1000:
            return 10**-5
        else:
            return 10**-6


    def train_model(self):
        input_size = 8 + 2 * self.days_length
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation='relu', input_dim=input_size),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(2)])
        #from keras.utils import multi_gpu_model
        #self.model_cases = multi_gpu_model(self.model_cases, gpus=2)

        batch_size = 10000
        epochs = 1000
        split = 0.2
        s_time = time.time()

        #callbacks = [tf.keras.callbacks.LearningRateScheduler(self.schedule, verbose=0)]
        adam = tf.keras.optimizers.Adam(learning_rate=.5*10 ** -5)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])


        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=split)  #, random_state=42)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,  validation_data=(X_test, y_test))  # callbacks=callbacks,
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Cases Test loss:', score[0])
        print('Cases Test accuracy:', score[1])
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_abs = np.abs(train_pred-y_train)
        test_abs = np.abs(test_pred-y_test)
        y_test[y_test == 0] = 10**50  # setting zero values to inf to get ignored
        y_train[y_train == 0] = 10**50

        print('Train mean absolute percentage error:', np.mean(train_abs/y_train))
        print('Test mean absolute percentage error:', np.mean(test_abs/y_test))
        #IPython.embed(header='test')


        print('Training Time', time.time() - s_time)
        IPython.embed(header='train')

        self.model.save('models/model.h5')

    def load_model(self):
        self.model = load_model('models/model.h5')



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

                return None, None, None, None

        population = self.country_dataset[index][2]/10**8  # 100 Millions
        density = self.country_dataset[index][5]/1000  # (1000 p/Km)
        area = self.country_dataset[index][6]/10**6  # ( Million Km)
        if self.country_dataset[index][10][:-1] == 'N.A':
            urban = 0.
        else:
            urban = float(self.country_dataset[index][10][:-1])/100  # to remove percentage sign (percentage out of one)

        return population, density, area, urban


    def predict(self):
        cases_factor = 1000
        deaths_factor = 100
        # predict for n days by appending each case,death prediction to the dataset
        dataset_cases = np.pad(self.dataset_cases, ((0, 0), (0, self.day_predict)), 'constant', constant_values=0)  # add zeros for new days to predict
        dataset_deaths = np.pad(self.dataset_deaths, ((0, 0), (0, self.day_predict)), 'constant', constant_values=0)
        start_index = len(self.dataset_cases[0])
        for index, row in enumerate(dataset_cases):
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

            for day in range(self.day_predict):
                start = day + start_index - self.days_length
                end = day + start_index
                # using total values
                #y_cases = [float(sum(row[4:i]))/1000 for i in range(start, end)]
                #y_deaths = [float(sum(self.dataset_deaths[index][4:i]))/100 for i in range(start, end)]  # note: output per 100 deaths
                # using new values
                y_cases = [(dataset_cases[index][i] - dataset_cases[index][i - 1]) / cases_factor for i in range(start, end)]
                y_deaths = [(dataset_deaths[index][i] - dataset_deaths[index][i - 1]) / deaths_factor for i in range(start, end)]


                days_since_n_cases = (start_index + day - case_index)/100
                X = [population, density, area, urban, days_since_n_cases, LockDown, longitude, latitude]
                X = np.array([X + list(y_cases) + list(y_deaths)])

                predict = self.model.predict(X)[0]
                #print('cases:{}, deaths:{}'.format(np.round(predict[0] * cases_factor), np.round(predict[1] * deaths_factor)))
                dataset_cases[index][start_index + day] = max(0, np.round(predict[0] * cases_factor)) + dataset_cases[index][start_index + day-1]
                dataset_deaths[index][start_index + day] = max(0, np.round(predict[1] * deaths_factor)) + dataset_deaths[index][start_index + day-1]

        self.save_prediction(dataset_cases, dataset_deaths)


    def save_prediction(self, dataset_cases, dataset_deaths):
        dim = len(dataset_cases[0])
        columns = ['Province/State', 'Country/Region', 'Lat', 'Long']
        date = datetime.datetime(2020, 1, 22)
        for i in range(dim - 4):
            columns.append(date.strftime("%d-%m-%Y"))
            date += datetime.timedelta(days=1)

        cases_filename = 'output/predict2_cases.csv'
        deaths_filename = 'output/predict2_deaths.csv'

        with open(cases_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in dataset_cases:
                writer.writerow(row)
        with open(deaths_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in dataset_deaths:
                writer.writerow(row)



if __name__ == '__main__':
    Training()

