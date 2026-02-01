# Import libraries/functions
from pygam import LinearGAM, s, f, l
import pandas as pd
import numpy as np 

#Import Data and Manipulate Data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["year"]  = data["Timestamp"].dt.year
data["month"] = data["Timestamp"].dt.month
data["day"]   = data["Timestamp"].dt.weekday   # day of week (0-6)
data["hour"]  = data["Timestamp"].dt.hour

x = data[['year', 'month', 'day', 'hour']].values
y = data['trips'].astype(float).values

#Create Model and Fit 
model = LinearGAM(
    s(0, n_splines=10) +  # year
    s(1, n_splines=12) +  # month
    f(2) +                # day of week
    s(3, n_splines=24)    # hour
)
modelFit = model.fit(x,y)

#Load Test Data
test = pd.read_csv(
    "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
)

test["Timestamp"] = pd.to_datetime(test["Timestamp"])
test["year"]  = test["Timestamp"].dt.year
test["month"] = test["Timestamp"].dt.month
test["day"]   = test["Timestamp"].dt.weekday
test["hour"]  = test["Timestamp"].dt.hour

x_test = test[["year", "month", "day", "hour"]].values

#Predict
pred = modelFit.predict(x_test)
pred = np.maximum(pred, 0)