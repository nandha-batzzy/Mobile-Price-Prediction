import numpy as np

from flask import Flask,request,render_template,jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_fea = [int(x) for x in request.form.values()]
    fin_features = [np.array(int_fea)]

    my_prediction = model.predict(fin_features)

    out = round(my_prediction[0],2)

    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)