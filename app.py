import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("Tips Prediction Project")

if "scaling_done" not in st.session_state:
    st.session_state.scaling_done = False

if "splitting_done" not in st.session_state:
    st.session_state.splitting_done = False

if "model_done" not in st.session_state:
    st.session_state.model_done = False

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write("Data Preview")
    st.write(data)

    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Column Names")
    st.write(data.columns)

    st.subheader("Missing Values")
    st.write(data.isna().sum())

    if data.isna().sum().sum() == 0:
        le_sex = LabelEncoder()
        le_smoker = LabelEncoder()
        le_day = LabelEncoder()
        le_time = LabelEncoder()

        data['sex'] = le_sex.fit_transform(data['sex'])
        data['smoker'] = le_smoker.fit_transform(data['smoker'])
        data['day'] = le_day.fit_transform(data['day'])
        data['time'] = le_time.fit_transform(data['time'])

        X = data.drop(['tip'], axis=1)
        Y = data[['tip']]

        st.subheader("Features (X)")
        st.write(X)
        st.subheader("Target (Y)")
        st.write(Y)

        if st.button("Proceed to Scaling"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.X_scaled = X_scaled
            st.session_state.X = X
            st.session_state.Y = Y
            st.session_state.scaling_done = True

    if st.session_state.scaling_done:
        st.subheader("Scaled Features")
        st.write(pd.DataFrame(st.session_state.X_scaled, columns=st.session_state.X.columns))

        if st.button("Split the Data"):
            X_train, X_test, Y_train, Y_test = train_test_split(
                st.session_state.X_scaled,
                st.session_state.Y,
                test_size=0.2,
                random_state=42
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.Y_train = Y_train
            st.session_state.Y_test = Y_test
            st.session_state.splitting_done = True

    if st.session_state.splitting_done:
        option = st.selectbox("Select to view shape", ["X_train", "X_test", "Y_train", "Y_test"])
        if option == "X_train":
            st.write(st.session_state.X_train.shape)
        elif option == "X_test":
            st.write(st.session_state.X_test.shape)
        elif option == "Y_train":
            st.write(st.session_state.Y_train.shape)
        elif option == "Y_test":
            st.write(st.session_state.Y_test.shape)

        if st.button("Train Linear Regression Model"):
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.Y_train)
            st.session_state.model = model
            y_pred = model.predict(st.session_state.X_test)
            st.session_state.y_pred = y_pred
            st.session_state.model_done = True

    if st.session_state.model_done:
        st.subheader("Predicted Tips")
        st.write(st.session_state.y_pred)

        if st.button("Show Evaluation Metrics"):
            mse = mean_squared_error(st.session_state.Y_test, st.session_state.y_pred)
            r2 = r2_score(st.session_state.Y_test, st.session_state.y_pred)
            st.write("Mean Squared Error:", mse)
            st.write("R2 Score:", r2)

        if st.button("Plot Actual vs Predicted"):
            plt.figure(figsize=(8, 5))
            plt.scatter(st.session_state.Y_test, st.session_state.y_pred, color='blue')
            plt.plot(st.session_state.y_pred,st.session_state.y_pred,color='red')
            plt.xlabel("Actual Tips")
            plt.ylabel("Predicted Tips")
            plt.title("Actual vs Predicted Tips")
            st.pyplot(plt)
