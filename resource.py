import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2

from scipy.integrate import odeint
from scipy.optimize import minimize


def predict_image(filepath):
    image = cv2.resize(plt.imread(filepath), (150, 150))
    image = (np.dstack([image, image, image]).astype('float32') / 255).reshape((1, 150, 150, 3))
    model = keras.models.load_model('.')
    return model.predict(image)[0, 0]


def solve_equation(SIR, time, population, beta, gamma):
    S, I, R = SIR
    return -beta * S * I / population, beta * S * I / population - gamma * I, gamma * I


def model(I_interval, R_interval, population, beta, gamma):
    S, I, R = odeint(solve_equation, (population - I_interval[0] - R_interval[0], I_interval[0],
                     R_interval[0]), np.linspace(0, len(I_interval), len(I_interval)),
                     args=(population, beta, gamma)).T
    return np.linalg.norm(np.diff(I_interval) - np.diff(I + R))


def make_frame(country_name, infected, recovered, deaths, smooth_window=3):
    country = pd.DataFrame(
        [infected.loc[country_name], recovered.loc[country_name], deaths.loc[country_name]]).T
    population = country.iloc[-1, 0]
    country = country.iloc[2:-1].reset_index()
    country.columns = ['Date', 'Infected', 'Recovered', 'Deaths']
    country['Removed'] = country['Recovered'] + country['Deaths']
    country["Date"] = pd.to_datetime(country["Date"], format="%m/%d/%y")
    for column in ['Infected', 'Recovered', 'Deaths', 'Removed']:
        country[column + "_Av"] = country[column].rolling(window=smooth_window).mean()
    return population, country


def compute_paramers(df, population, start_index, gamma=1/30, ndays=8):
    for i in range(start_index, len(df) - ndays):
        df.loc[i, 'Beta'] = minimize(lambda x: model(df['Infected_Av'][i:i + ndays].to_numpy(),
                                                     df['Removed_Av'][i:i + ndays].to_numpy(),
                                                     population, x, gamma),
                                     x0=0.5, method='powell').x
        df.loc[i, 'Gamma'] = gamma
    return df


def analyze(country_name, infected, recovered, deaths):
    population, df = make_frame(country_name, infected, recovered, deaths)
    df = compute_paramers(df, population, df[df['Infected_Av'] > 1000].index[0])
    df['Rt'] = df['Beta'] / df['Gamma']
    return population, df.iloc[df[df['Infected_Av'] > 1000].index[0]:]




