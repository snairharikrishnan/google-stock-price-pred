import io
import pandas as pd
import numpy as np
from flask import Flask, request,render_template,send_file
import matplotlib.pyplot as plt

app=Flask(__name__)

real_stock=pd.read_csv('C:/Users/snair/Documents/Data Science Assignment/Data Sets/LSTM/Google_Stock_Price_Test.csv')
real_stock=real_stock.iloc[:,1:2].values #converting to array
pred_stock=np.load("predicted_stock.npy")

dates=pd.date_range('2017-01-03', '2017-01-31').to_series()
dayno=pd.DataFrame(dates.dt.dayofweek,columns=['daynumber']).reset_index()
dayno=dayno[(dayno.daynumber!=5) & (dayno.daynumber!=6)]#removing weekends
dayno=dayno.drop(index=13).reset_index()#dropping as absent in dataset

@app.route('/forecast',methods=['POST'])
def forecast():
    days = int(request.form.get("days"))#accepting no of days to be forecasted    
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(dayno.loc[:days-1,'index'],real_stock[:days],color='blue',label='Real Stock Price',)
    plt.plot(dayno.loc[:days-1,'index'],pred_stock[:days],color='red',label='Predicted Stock Price')
    plt.xticks(rotation='vertical')
    fig.suptitle('Google Stock Price', fontsize=25)
    plt.xlabel('Date',fontsize=15)
    plt.ylabel('Stock Price',fontsize=15)
    plt.legend()
    
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/')
def home():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)