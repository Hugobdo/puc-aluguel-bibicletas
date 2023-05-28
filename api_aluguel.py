from fastapi import FastAPI
import uvicorn
import ast
from ml import predict
import json
import pickle
from catboost import CatBoostRegressor

app = FastAPI()
model = CatBoostRegressor().load_model(fname='./extras'
                '/models/aluguel_bicicletas_catboost', format="cbm")
sc = pickle.load(open('./extras/scaler/std_scaler.pkl', 'rb'))
with open('./extras/dummies/dummies') as handler:
    dummies = json.loads(handler.read())

@app.get("/pred/cat/{dados_entrada_modelo}")
async def predict_model(dados_entrada_modelo: str):
    dict_dados = ast.literal_eval(dados_entrada_modelo)
    return {"dados": dict_dados, "resultado_pred": predict.predict(dict_dados, model, sc, dummies)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

'''
Exemplo de chamada para teste:
{
    "datetime": "2011-01-20 00:00:00",
    "season": 1,
    "holiday": 0,
    "workingday": 1,
    "weather": 1,
    "temp": 10.66,
    "atemp": 11.365,
    "humidity": 56,
    "windspeed": 26.0027,
    "Weather": 1,
    "Temperature": 10.66,
    "Humidity": 56,
    "Wind_Speed": 26.0027,
    "Seasons": 1,
    "Holiday": 0,
    "WorkingDay": 1
}
'''