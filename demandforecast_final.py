"""# This Deep learning model is Demand forecasting model which uses the Electricity consumption dataset flow 
# to forecast the electricity requirements further on time series data.
# This model uses Tensorflow (Keras) libraries to build the model.
# We have used LSTM(Long-Short Term Memory) algorithm for this predic"""

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers.recurrent import LSTM 
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential


"""#Loading the Dataset with consideration of First Column. """
demand_dataset = pd.read_csv("/home/arishti/Inteliment/Supply Chain Management/Demand Forecasting/UsingTimeSeries/Datasets/dados_Brazil_GDP_Electricity.csv", index_col=1)


"""#dropping the unnecessary columns which are not considered for the forecasting. """
demand_dataset = demand_dataset.drop(demand_dataset.columns[0], axis=1)


"""# visualizing and plotting the datasets."""
demand_dataset.plot()
plt.show()

"""#Visualizing and Plotting using the Scatter plots for the electricity consumption in a particular year in country."""
plt.scatter(x=demand_dataset.iloc[:,0], y=demand_dataset.iloc[:,1])
_=plt.show()


"""#MinMaxScaler is generally used for the normalizing the data and bringing the datavalues in the range of (0,1)."""
data = demand_dataset.iloc[:,0].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

"""#Fit to emand_dataset, then transform it."""
data = scaler.fit_transform(data)


NUM_TIMESTEPS = 5   #Number of dimensional Data being generated in arrays.
max_elements = len(data) - NUM_TIMESTEPS - 1
X = np.zeros((data.shape[0], NUM_TIMESTEPS))
Y = np.zeros((data.shape[0], 1))
for i in range(len(data) - NUM_TIMESTEPS - 1):
    X[i] = data[i:i + NUM_TIMESTEPS].T  # Saving all the X data and preparing for Training & Testing dataset 
    Y[i] = data[i + NUM_TIMESTEPS + 1]  # Saving all the Y data and preparing for Training & Testing dataset 
    #print("X value : ",X[i])
    #print("Y Value : ",Y[i])
    
"""reshape X to three dimensions (samples, timesteps, features)"""
X = np.expand_dims(X, axis=2)

#X = X.reshape(X.shape[0], X.shape[1], 1)
print("Value of X is ")
print(X)
X = X[:max_elements]
Y = Y[:max_elements]


"""Spliting the data in Train, Test partitions with 70-30%"""
sp = int(0.7 * len(data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)


"""Epochs is the number times that the learning algorithm will work through the entire training dataset."""
NUM_EPOCHS = 200   

"""Batch Size is the number of samples to work through before updating the internal model parameters."""
BATCH_SIZE = 5     

np.random.seed(123456) 

 
"""A user-defined function which uses the LSTM model and complies with the ADAM Optimizer.
ADAM Optimizer computes individual learning rates for different parameters"""

def build_model_stateless(): 
    model = Sequential()
    model.add(LSTM(5, input_shape=(NUM_TIMESTEPS, 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model
    

model_stateless = build_model_stateless()



""" 
EarlyStopping - Stop training when a monitored quantity has stopped improving.
Patience - Number of epochs with no improvement after which training will be stopped.
"""
early_stopping = EarlyStopping(patience=4)
history = model_stateless.fit(Xtrain, Ytrain, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping], verbose=0)


"""
Ploting the predicted and Actual Values 
"""
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
_=plt.show()

"""
Testing the Test dataset on the trained model to check the accuracy and the values
"""
test_length = NUM_TIMESTEPS


"""Considering the train dataset and creating the split between the train and test"""
dataset_train = demand_dataset.iloc[0:-test_length,:] 



last_point = dataset_train.iloc[-NUM_TIMESTEPS:, 0].values.reshape(1, NUM_TIMESTEPS)

""" Transforming the last point values in the range of 0,1 """
last_point_scaled = scaler.transform(last_point)
last_point_scaled = last_point_scaled.reshape(1, NUM_TIMESTEPS, 1)


"""Calculating the total number of Test datapoints"""
num_predictions = test_length + 10
year_ini = dataset_train.index[-1] + 1 


last_x = last_point_scaled
predictions = []
years = []
for i in range(num_predictions):
    pred_tmp = model_stateless.predict(last_x)
    pred = scaler.inverse_transform(pred_tmp) #Inverse_transform will again bring back the 0,1 rangeining values to normal forms
    predictions.append(pred[0][0]) #appendin the results back to the predicting list 
    years.append(year_ini+i)
    next_x = np.roll(last_x, 1)
    next_x[0,0] = pred_tmp
    last_x = next_x

predictions_df = pd.DataFrame({'Forecast':predictions})   

ax = demand_dataset.iloc[:,0].plot()
predictions_df.plot(ax=ax)
plt.ylabel("Electricity")
plt.show()

"""printing the predicting values"""
for j in predictions:
    print(j)
    
demand_dataset.iloc[-test_length:, 0].values

pred_errors = demand_dataset.iloc[-test_length:, 0].values - predictions_df.iloc[:test_length,0].values

error_rate = np.sqrt(np.mean(pred_errors**2))

print("The RMSE Error is :" ,np.sqrt(np.mean(pred_errors**2)))
print("Accuracy rate is",(100-error_rate))