from flask import Flask, render_template,request,redirect,url_for
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("LinerReg.pkl",'rb'))
car =  pd.read_csv("D:\car_pred\clean_data.csv")


@app.route('/', methods = ['GET','POST'])
@app.route("/index", methods = ['GET','POST'])
def index():

    if request.method =="POST":
        name = request.form['carname']
        company = request.form['carcompany']
        year = request.form['year']
        kms_driven = request.form['km']
        fuel_type = request.form['fuel']

        prediction = model.predict(pd.DataFrame([[name,company,year,kms_driven,fuel_type]], columns = ['name','company','year','kms_driven','fuel_type']))
        price = str(np.round(prediction[0],2))
        #return str(prediction[0])
        print(price)

        return render_template("index.html",price=price)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug = True)
