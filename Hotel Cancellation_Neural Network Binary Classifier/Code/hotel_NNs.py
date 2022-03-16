# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:34:04 2021

@author: Neda
"""

# Build a NN model to predict hotel cancellations with a binary classifier.


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# load the Hotel Cancellations dataset
hotel = pd.read_csv('E:/OneDrive/Neda/Machine learning/Kaggle/Hotel/Data/hotel.csv')

# Create target
X = hotel.copy()
y = X.pop('is_canceled')


# Classes distribution
fig = plt.figure(1)
sns.countplot(y)

# X['arrival_date_month'] = \
#     X['arrival_date_month'].map(
#         {'January':1, 'February': 2, 'March':3,
#          'April':4, 'May':5, 'June':6, 'July':7,
#          'August':8, 'September':9, 'October':10,
#          'November':11, 'December':12}
#     )



# Numerical features
features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
    
    "booking_changes", "days_in_waiting_list"
     
]


# Categorical features
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
    
    "arrival_date_year", "assigned_room_type", 
]

# We didn't consider 5 features: 
# "agent" and "company", because they have high number of missing values.
# "country", because it has high cardinality and also has missing values  
# "reservation_status", because it is similar to the target "is-cancelled"
# "reservation_status_date", because it is a feature recorded after the target recorded. 


# Making transfrmer using pipeline for Imputation and Standardization of numerical features
transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)

# Making transformer using pipeline for Imputation and OneHotEncoding of categorical features
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)


# Making preprocessor by apply transformers to the corresponding columns
preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)


# train test split
# stratify to make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

# Fit the preprocessor to X_train and transform X_train and X_test 
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

# NN model for binary classification
# The model will have both batch normalization and dropout layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(units=256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(units=256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(units=1, activation='sigmoid')
])

# model for binary classification
model.compile(optimizer='Adam', 
              loss='binary_crossentropy', metrics=['binary_accuracy'])

# early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

# fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

# plot loss and accuracy over the epochs
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy loss")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


