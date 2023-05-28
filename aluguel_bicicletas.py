import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
import pickle
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings("ignore")

# Print helper function
def print_progress(message):
    print(f"=== {message} ===")

# Load data
def load_data(train_file, test_file):
    print_progress("Loading data")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    return df_train, df_test

# Preprocessing
def preprocess_data(df_train, df_test):
    print_progress("Preprocessing data")
    anosDict = {2011: 0, 2012: 1}
    date_format = '%Y-%m-%d %H:%M:%S'

    # Convert datetime column to datetime format
    df_train['Date'] = pd.to_datetime(df_train['datetime'], format=date_format)
    df_test['Date'] = pd.to_datetime(df_test['datetime'], format=date_format)

    # Extract features from datetime column
    df_train['Year'] = df_train['Date'].dt.year.map(anosDict)
    df_train['Month'] = df_train['Date'].dt.month
    df_train['Day'] = df_train['Date'].dt.day
    df_train['Hour'] = df_train['Date'].dt.hour
    df_train['DayName'] = df_train['Date'].dt.day_name()
    df_train['DayNumber'] = df_train['Date'].dt.dayofweek
    df_train.drop(columns='Date', inplace=True)

    df_test['Year'] = df_test['Date'].dt.year.map(anosDict)
    df_test['Month'] = df_test['Date'].dt.month
    df_test['Day'] = df_test['Date'].dt.day
    df_test['Hour'] = df_test['Date'].dt.hour
    df_test['DayName'] = df_test['Date'].dt.day_name()
    df_test['DayNumber'] = df_test['Date'].dt.dayofweek
    df_test.drop(columns='Date', inplace=True)

    # Rename columns
    df_train.rename(columns={'season': 'Seasons',
                             'holiday': 'Holiday',
                             'humidity': 'Humidity',
                             'windspeed': 'Wind_Speed',
                             'weather': 'Weather',
                             'atemp': 'aTemperature',
                             'temp': 'Temperature',
                             'casual': 'Casual',
                             'registered': 'Registered',
                             'workingday': 'WorkingDay'}, inplace=True)
    df_test.rename(columns={'season': 'Seasons',
                            'holiday': 'Holiday',
                            'humidity': 'Humidity',
                            'windspeed': 'Wind_Speed',
                            'weather': 'Weather',
                            'atemp': 'aTemperature',
                            'temp': 'Temperature',
                            'workingday': 'WorkingDay'}, inplace=True)

    # Reorder columns
    columns_order = ['datetime', 'Year', 'Month', 'Day', 'DayName', 'DayNumber', 'Hour', 'Weather', 'Temperature',
                     'count', 'Humidity', 'Wind_Speed', 'Seasons', 'Holiday', 'WorkingDay', 'Casual', 'Registered']
    df_train = df_train[columns_order]
    df_test = df_test[['datetime', 'Year', 'Month', 'Day', 'DayName', 'DayNumber', 'Hour', 'Weather', 'Temperature',
                       'Humidity', 'Wind_Speed', 'Seasons', 'Holiday', 'WorkingDay']]

    # Manipulate features
    df_train['Temperature'] = np.floor(df_train['Temperature']).astype(int)
    df_test['Temperature'] = np.floor(df_test['Temperature']).astype(int)

    return df_train, df_test

# Feature engineering

def feature_engineering(df_train, df_test):
    print_progress("Feature engineering")

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

    def normalize(df):
        result = df.copy()
        for attribute in df.columns:
            max_value = df[attribute].max()
            min_value = df[attribute].min()
            result[attribute] = (df[attribute] - min_value) / (max_value - min_value)
        return result

    df_train['Hour_Sin'] = np.sin(df_train['Hour'] * (2. * np.pi / 24))
    df_train['Hour_Cos'] = np.cos(df_train['Hour'] * (2. * np.pi / 24))
    df_train['DayName_Sin'] = np.sin(pd.Categorical(df_train['DayName']).codes * (2. * np.pi / 7))
    df_train['DayName_Cos'] = np.cos(pd.Categorical(df_train['DayName']).codes * (2. * np.pi / 7))
    df_train['Day_Period'] = df_train['Hour'].apply(extrair_periodo_do_dia)
    df_train['Rush_Hour'] = df_train['Hour'].apply(verifica_hora_de_pico)

    df_test['Hour_Sin'] = np.sin(df_test['Hour'] * (2. * np.pi / 24))
    df_test['Hour_Cos'] = np.cos(df_test['Hour'] * (2. * np.pi / 24))
    df_test['DayName_Sin'] = np.sin(pd.Categorical(df_test['DayName']).codes * (2. * np.pi / 7))
    df_test['DayName_Cos'] = np.cos(pd.Categorical(df_test['DayName']).codes * (2. * np.pi / 7))
    df_test['Day_Period'] = df_test['Hour'].apply(extrair_periodo_do_dia)
    df_test['Rush_Hour'] = df_test['Hour'].apply(verifica_hora_de_pico)

    categorical_cols = ['DayName', 'Day_Period']  # Add 'Day_Period' to categorical columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_train = pd.DataFrame(encoder.fit_transform(df_train[categorical_cols]))
    encoded_test = pd.DataFrame(encoder.transform(df_test[categorical_cols]))
    encoded_train.columns = encoder.get_feature_names(categorical_cols)
    encoded_test.columns = encoder.get_feature_names(categorical_cols)
    df_train = pd.concat([df_train, encoded_train], axis=1)
    df_test = pd.concat([df_test, encoded_test], axis=1)
    df_train.drop(categorical_cols, axis=1, inplace=True)
    df_test.drop(categorical_cols, axis=1, inplace=True)

    scaler = StandardScaler()
    X_train = df_train.drop(columns=['datetime', 'count','Casual', 'Registered'])
    X_test = df_test.drop(columns=['datetime'])

    # Reorder columns in X_train and X_test to match the expected order
    column_order = X_train.columns.tolist()
    X_train = X_train[column_order]
    X_test = X_test[column_order]

    # Normalize the data
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    return X_train, X_test

# Select top 3 models based on RMSLE
def select_top_models(X_train, y_train, X_val, y_val):
    print_progress("Selecting top models")
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "BaggingRegressor": BaggingRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "CatBoostRegressor": CatBoostRegressor(verbose=False),
        "LGBMRegressor": LGBMRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "XGBRegressor": XGBRegressor()
    }

    rmsle_scores = {}
    for name, model in models.items():
        print(f"Training model: {name}")
        model.fit(X_train, y_train)
        print(f"Training score: {model.score(X_train, y_train)}")
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        rmsle_train = rmsle(y_train, abs(y_train_pred))
        rmsle_val = rmsle(y_val, abs(y_val_pred))
        print(f"RMSLE on train set: {rmsle_train}")
        print(f"RMSLE on validation set: {rmsle_val}")
        print("==============================================")
        rmsle_scores[name] = rmsle_val

    # Select top 3 models with lowest RMSLE
    top_models = sorted(rmsle_scores, key=rmsle_scores.get)[:3]
    models = {k: models[k] for k in top_models}

    return models

def grid_search_models(models, X_train, y_train):
    print_progress("Grid search on top models")
    
    best_model = None
    best_score = float('-inf')
    best_params = None
    
    for name in models:
        model = models[name]
        print(f"Model: {name}")

        if name == "CatBoostRegressor":
            params = {'iterations': [100, 200],
                        'learning_rate': [0.01, 0.05],
                        'depth': [4, 6, 8],
                        'l2_leaf_reg': [1, 3]}

        elif name == "RandomForestRegressor":
            params = {'n_estimators': [100, 200],
                        'max_depth': [4, 6],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]}

        elif name == "BaggingRegressor":
            params = {'n_estimators': [100, 200],
                        'max_samples': [0.1, 0.5],
                        'max_features': [0.1, 0.5]}
        else:
            params = {}

        for param_name, param_values in params.items():
            print(f"Parameter: {param_name}")
            param_grid = {param_name: param_values}
            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_log_error', cv=5, verbose=1)
            grid.fit(X_train, y_train)
            print(f"Best score: {grid.best_score_}")
            print(f"Best params: {grid.best_params_}")
            print("==============================================")

            if grid.best_score_ > best_score:
                best_model = grid.best_estimator_
                best_score = grid.best_score_
                best_params = grid.best_params_

    return best_model, best_score, best_params

# Root Mean Squared Logarithmic Error
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Prediction
def predict(models, X_test):
    print_progress("Prediction")
    predictions = []
    for name, model in models.items():
        print(f"Model: {name}")
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    return predictions

# Save predictions to file
def save_predictions(predictions, df_test):
    print_progress("Saving predictions")
    datetime = df_test['datetime']
    df_predictions = pd.DataFrame({'datetime': datetime, 'count': np.floor(np.exp(predictions)).astype(int)})
    df_predictions.to_csv('submission.csv', index=False)

# Save models
def save_models(models):
    print_progress("Saving models")
    for name, model in models.items():
        model.save_model(f"model_{name}.cbm", format="cbm")

# Load models
def load_models():
    print_progress("Loading models")
    models = {}
    for name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "KNeighborsRegressor", "DecisionTreeRegressor",
                 "RandomForestRegressor", "BaggingRegressor", "AdaBoostRegressor", "CatBoostRegressor",
                 "LGBMRegressor", "GradientBoostingRegressor", "XGBRegressor"]:
        model = CatBoostRegressor().load_model(f"model_{name}.cbm")
        models[name] = model
    return models

# Load scaler
def load_scaler():
    print_progress("Loading scaler")
    scaler = pickle.load(open('std_scaler.pkl', 'rb'))
    return scaler

train_file = 'train.csv'
test_file = 'test.csv'

# Load data
df_train, df_test = load_data(train_file, test_file)

# Preprocess data
df_train, df_test = preprocess_data(df_train, df_test)

# Feature engineering
X_train, X_test = feature_engineering(df_train, df_test)

# Split the training data into train and validation sets
y_train = df_train['count']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Select top 3 models
top_models = select_top_models(X_train, y_train, X_val, y_val)

# Perform grid search on top models
best_model, best_score, params = grid_search_models(top_models, X_train, y_train)

# Train the best model
best_model.set_params(**params)
best_model.fit(X_train, y_train)

# Save the best model
pickle.dump(best_model, open(r'model/best_model.pkl', 'wb'))

# Load the best model
best_model = pickle.load(open(r'model/best_model.pkl', 'rb'))

# Make predictions
y_pred = best_model.predict(X_test[0:1])
