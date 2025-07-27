
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Activer CORS pour permettre les requêtes depuis Flutter

# Charger les variables d'environnement
load_dotenv()

# Charger les modèles et leurs scalers
models = {
    'random_forest': {
        'model': joblib.load('models/random_forest_model.pkl'),
        'scaler': joblib.load('models/random_forest_scaler.pkl')
    },
    'xgboost': {
        'model': joblib.load('models/xgboost_model.pkl'),
        'scaler': joblib.load('models/xgboost_scaler.pkl')
    },
    'catboost': {
        'model': joblib.load('models/catboost_model.pkl'),
        'scaler': joblib.load('models/catboost_scaler.pkl')
    },
    'svr': {
        'model': joblib.load('models/svr_model.pkl'),
        'scaler': joblib.load('models/svr_scaler.pkl')
    },
    'lightgbm': {
        'model': joblib.load('models/lightgbm_model.pkl'),
        'scaler': joblib.load('models/lightgbm_scaler.pkl')
    }
}

# Modèle mathématique (adapté à votre cas, à ajuster selon votre équation réelle)
def mathematical_model(inputs):
    # Exemple : production = irradiation * efficiency * area
    irradiation = inputs.get('irradiation', 0)
    efficiency = 0.2  # Valeur par défaut, ajustez selon votre modèle
    area = 1.0  # Valeur par défaut, ajustez selon votre modèle
    return irradiation * efficiency * area

# Endpoint pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON envoyées par Flutter
        data = request.get_json()

        # Vérifier les caractéristiques attendues
        required_features = [
            'temperateur moyenne', 'temperateur maximale', 'temperateur minimale(°c)',
            'vitesse de vent', 'pression', 'Sunrise', 'Sunset', 'Day length', 'irradiation'
        ]
        features = []
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing feature: {feature}'
                }), 400
            features.append(data[feature])

        # Convertir en array numpy et reshaper pour la prédiction
        features = np.array([features])

        # Calculer les prédictions pour chaque modèle
        predictions = {}
        for model_name, model_data in models.items():
            scaled_features = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(scaled_features)[0]
            predictions[model_name] = float(prediction)

        # Ajouter la prédiction du modèle mathématique
        predictions['mathematical'] = mathematical_model(data)

        # Sélectionner le meilleur résultat (par exemple, la valeur maximale)
        best_model = max(predictions, key=predictions.get)
        best_prediction = predictions[best_model]

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'best_model': best_model,
            'best_prediction': best_prediction
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint de test
@app.route('/')
def home():
    return jsonify({'message': 'API for PV estimation is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
