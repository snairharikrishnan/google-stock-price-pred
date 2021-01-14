import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_dataset=pd.read_csv('C:/Users/snair/Documents/Data Science Assignment/Data Sets/LSTM/Google_Stock_Price_Train.csv')
train_data=train_dataset.iloc[:,1:2].values #converting to array


#Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_data=sc.fit_transform(train_data)


# Taking the window size as 60 days
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i])
    
x_train=np.array(x_train)
y_train=np.array(y_train)

x_train.shape
#Reshape to be RNN input
x_train=np.reshape(x_train,(1198,60,1))

from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Bidirectional

model=Sequential()
model.add(Bidirectional(LSTM(units=50,return_sequences=True,input_shape=(60,1))))
#model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=50,return_sequences=True)))
#model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=50,return_sequences=True)))
#model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=50)))
#model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,epochs=100,batch_size=32)

test_dataset=pd.read_csv('C:/Users/snair/Documents/Data Science Assignment/Data Sets/LSTM/Google_Stock_Price_Test.csv')
real_data=test_dataset.iloc[:,1:2].values #converting to array

complete_dataset=pd.concat((train_dataset['Open'],test_dataset['Open']),axis=0)

inputs=complete_dataset[len(complete_dataset)-len(test_dataset)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

x_test=np.array(x_test)    
x_test=np.reshape(x_test,(20,60,1))

pred=model.predict(x_test)
pred=sc.inverse_transform(pred)
np.sqrt(np.mean((pred-real_data)**2)) #RMSE = 11.53

np.save("predicted_stock",pred)

plt.plot(real_data,color='red',label='Real Stock Price')
plt.plot(pred,color='blue',label='Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()








