import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

appl = Flask(__name__)
from statsmodels.iolib.smpickle import load_pickle
model = load_pickle("slr_mnc.pkl")

@appl.route('/')
def home():
    return render_template('index.html')

@appl.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    nifty = float_features[0];
    nifty_sq = nifty*nifty
    mnc = pd.DataFrame([[nifty, nifty_sq]], columns=["Nifty50", "Nifty50_sq"])
    x = model.predict(mnc)
    print(float(round(x,2)))
#    flt_features = [float(x) for x in request.form.values()]
##    int_features = [int(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    prediction = model.predict(final_features)
#
#    output = round(prediction[0], 2)
#
#    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('index.html', prediction_text= "MNC NAV is {}".format(float(round(x,2))))


if __name__ == "__main__":
    appl.run(debug=True)