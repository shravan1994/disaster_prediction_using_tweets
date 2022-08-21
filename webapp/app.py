from flask import Flask, render_template, request
from model import model_predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict")
def predict():
    tweet = request.args.get('tweet')
    target_label = model_predict(tweet)

    params = {
        'result': 'This is real disaster' if target_label == 1 else 'This is not a real disaster'
    }
    return render_template('home.html', params=params)
