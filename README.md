# Google Stock Price Prediction
## Overview
Stock market is a highly volatile market that requires constant attention without which there would be financial losses. Predicting prices of stocks helps traders to take immediate investment decisions. One of the main aims of a trader is to predict the stock price such that he can sell it before its value decline, or buy the stock before the price rises. Here we are predicting the stock prices of Google for January 2017 using previous stock prices.
## Demo
Link: https://google-stock-price-prediction.herokuapp.com/

![](/static/demo.JPG)

 ## Key Highlights
 * Data collected from Kaggle
 * Normalized the data
 * A 60-day window was taken for each training data as the independent variables and the 61st day as the dependent variable
 * Deep Leaning model based on Bidirectional LSTM was designed and trained on the training data
 * Predicted stock prices were plotted against actual stock prices
 * Flask used as the web application framework
 * Accepted the number of business days to be forecasted from the user
 * The stock prices for those days were predicted and plotted
 * The graph was passed to the web page for display
 * Web page built using HTML and CSS
 * Deployment done using Heroku

![](/static/forecast.JPG)

## Built Using
<img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width=280> <img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=180> <img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" width=180>  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width=80>
