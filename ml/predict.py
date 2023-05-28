import pandas as pd
import numpy as np


def extrair_periodo_do_dia(hora):
    if hora in range(12):
        return 'Morning'
    elif hora in range(12, 18):
        return 'Afternoon'
    elif hora in range(18, 22):
        return 'Evening'
    else:
        return 'Night'


def verifica_hora_de_pico(hora):
    return 0 if hora in [0, 1, 2, 3, 4, 5, 6, 10, 21, 22, 23] else 1

def predict(data, model, sc, dummies):
    df = pd.DataFrame([data])

    datetime = df['datetime']

    df['Date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df['Year'] = df['Date'].dt.year
    df['Year'] = df['Year'].map({2011:0,2012:1})
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['DayName'] = df['Date'].dt.day_name()
    df['DayNumber'] = df['Date'].dt.dayofweek
    df.drop(columns='Date', inplace=True)

    df = df[['datetime', 'Year', 'Month',
             'Day', 'DayName', 'DayNumber',
             'Hour', 'Weather', 'Temperature',
             'Humidity', 'Wind_Speed', 'Seasons', 'Holiday', 'WorkingDay']]

    df['Hour_Sin'] = np.sin(df['Hour'] * (2. * np.pi/24))
    df['Hour_Cos'] = np.cos(df['Hour'] * (2. * np.pi/24))

    df['DayName_Sin'] = np.sin(pd.Categorical(df['DayName']).codes * (2. * np.pi/7))
    df['Dayname_Cos'] = np.cos(pd.Categorical(df['DayName']).codes * (2. * np.pi/7))

    df['Day_Period'] = df['Hour'].apply(extrair_periodo_do_dia)
    df['Rush_Hour'] = df['Hour'].apply(verifica_hora_de_pico)

    for period in dummies['Day_Period']:
        df['Day_Period_' + period] = np.where(df['Day_Period'] == period, 1, 0)

    df.drop(columns=['datetime', 'Day_Period', 'DayName'], inplace=True)

    data_predict = sc.transform(df)
    pred = model.predict(data_predict)
    df['count'] = np.floor(np.exp(pred)).astype(int)

    df_final = pd.concat([df['count'], datetime], axis=1)

    return df_final.to_dict('records')

