import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Charger le modèle à partir du fichier
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données de la requête
    data = request.get_json(force=True)

    # Extraire les valeurs
    RM = data['RM']
    PTRATIO = data['PTRATIO']
    LSTAT = data['LSTAT']

    # Faire la prédiction
    prediction = loaded_model.predict([[RM, PTRATIO, LSTAT]])

    # Retourner la prédiction
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
