import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
import re
from sklearn.metrics import accuracy_score

data=pd.read_csv('Airline_review.csv')
print(data.columns)

#checking missing values
print(data.isna().sum())

#dropping missing values within the aircraft, type of traveler, seat type, route and date flown
data=data.dropna(subset=['Aircraft','Type Of Traveller','Seat Type','Route','Date Flown'])

#Filling NAs with means grouped by seat type and aircraft
data[['Seat Comfort', 'Cabin Staff Service','Food & Beverages','Ground Service','Inflight Entertainment','Wifi & Connectivity']] = data.groupby(['Airline Name', 'Seat Type'])[['Seat Comfort', 'Cabin Staff Service','Food & Beverages','Ground Service','Inflight Entertainment','Wifi & Connectivity']].transform(lambda x: x.fillna(x.mean()))
data['Overall_Rating']=pd.to_numeric(data['Overall_Rating'], errors='coerce')
#dropping any remaining NAs
data=data.dropna()

#checking the data set overview
#no null values with 20 attributes and 5946 observations
print(data.shape)
print(data.dtypes)

#eliminating anything that is not character from the Destination, Origin and Transit columns
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text) 

#Splitting the Route column into Origin and Destination and Transit
def split_route(route):
    parts=route.split(' to ',1)

    if len(parts)<2:
        return None,None,None
    Origin=clean_text(parts[0].strip())
    Destination=clean_text(parts[1].strip())

    if 'via' in Destination:
        Destination, Transit=Destination.split(' via ',1)
        Transit=clean_text(Transit.strip())
    else:
        Transit=None
    return Origin, Destination, Transit

data[['Origin', 'Destination', 'Transit']] = data['Route'].apply(lambda x: pd.Series(split_route(x)))

#separating the target variable to the rest of the data
X=data.drop(columns=['Unnamed: 0','Review_Title','Review Date','Verified','Review','Date Flown','Recommended','Route','Aircraft','Destination','Origin','Transit'])
y=data['Recommended']
y = y.map({'yes': 1, 'no': 0})

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)

#Encoding Aircraft, Type of Traveller, Seat Type and Route
#Discovering unique values for the ordinal encoder column
print(data['Type Of Traveller'].unique())
print(data['Seat Type'].unique())

#creating the categories for the Seat Type
categories_seat=[['Economy Class','Premium Economy','Business Class','First Class']]

transformer=ColumnTransformer([('ordinal',OrdinalEncoder(categories=categories_seat),['Seat Type']),
                                ('one_hot',OneHotEncoder(handle_unknown='ignore'),['Airline Name','Type Of Traveller'])], remainder='passthrough', verbose_feature_names_out=False)


steps=[('transform',transformer),
        ('scaler',StandardScaler(with_mean=False)),
        ('model',RandomForestClassifier(random_state=42,max_depth=None,min_samples_leaf=1,min_samples_split=10,n_estimators=200))]

pipe=Pipeline(steps)
pipe.fit(X_train,y_train)
joblib.dump(pipe,'airline_review_model.joblib')
print('Model saved successfully')