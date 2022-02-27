#API FLASK run (commande : python api/api.py)
# Local Adresse :  http://127.0.0.1:5000/credit/IDclient
# adresse distance : https://api-prediction-credit.herokuapp.com/credit/idclient
# Github depo : https://github.com/DeepScienceData/API-Prediction

from flask import Flask
from flask import jsonify
import pandas as pd
import numpy as np
from joblib import load
import os


app = Flask(__name__)

clf = load("best_model_lgbm.joblib")

sample = pd.read_csv('X_test_sample700.csv', index_col='SK_ID_CURR', encoding ='utf-8')

sample.head()

@app.route('/')

def home():

    return jsonify(username='GassMamadou' , email='birro90@hotmail.fr')
    

@app.route('/credit/<int:id_client>' , methods=['GET'])

def credit(id_client):
    
        id = id_client

        score = clf.predict_proba(sample.loc[[id]])[:,1]

        predict = clf.predict(sample.loc[[id]])

        # round the predict proba value and set to new variable
        percent_score = score*100
               
        id_risk = np.round(percent_score, 3)
        
        # create JSON object
        output = {'prediction': int(predict), 'client risk in %': float(id_risk)}

        print('Nouvelle Prédiction : \n', output)
        
        return jsonify(output)
        
if __name__ == '__main__':

    port = os.environ.get("PORT", 5000)
    app.run(host='0.0.0.0', port=port,debug=True)# debug=True