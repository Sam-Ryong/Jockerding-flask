from flask import Flask, request, jsonify


app = Flask(__name__)


# Load the model


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'msg_helmet': 'hello.'})


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
