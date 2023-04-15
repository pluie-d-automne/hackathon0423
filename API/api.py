from flask import Flask, request
from flask_restful import Api, Resource
from tensorflow import keras
import numpy as np
app = Flask(__name__)
api = Api()

model = keras.models.load_model('CNN_model.h5')

@app.route('/api/model', methods=['POST'])
def get(): 
    inp = request.get_json()
    input = inp['input']
    example = input_to_array(input)
    prediction = model.predict(example, verbose=0)
    prediction = np.where(prediction[0]==max(prediction[0]))[0][0]
    return str(prediction)
    
def input_to_array(input):
    X = []
    for line in input:
        box = np.zeros((5))
        for i in range(len(line)):
            box[i] = float(line[i])
        X.append(box)
    return np.array([np.array(X)])

api.init_app(app)
if __name__ == '__main__':
    app.run(debug=True, port=3000, host="127.0.0.1")