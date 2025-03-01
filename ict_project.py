import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pickle
from keras.models import load_model

from stock_data import preprocess



import os
  

# Setting the title of the webpage
st.title("Stock Price Pattern Prediction")

# Display today's date in the webpage
today = datetime.today().strftime('%Y-%m-%d')
st.write(f"### Today's Date: {today}")

#Heading for he next 15 days prediction
st.subheader("Predict Next 15 Days Stock Price Pattern")

#Taking the name of the stock as input
stock_name = st.text_input("Enter Stock Name:")

#Load the scalers and Models
    #Loading 2 scalers for scaling the data
with open(r"Models\scaler_gru.pkl", 'rb') as f:
    market_data_scaler = pickle.load(f)

with open(r"Models\scaler_close_price.pkl", 'rb') as f:
    close_price_scaler = pickle.load(f)

    #Loading the models
market_model = load_model(r"Models\gru_model_last.keras")
close_price_model = load_model(r"Models\close_price_model_last.keras")

prediction_days = 15

#Now getting the past 60 days stock data for prediction
def get_stock_data(stock_name):
    # end_date = datetime.today()
    # start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    ticker = yf.Ticker(stock_name)
    data = ticker.history(period = 'max', interval = '1d')
    print(data)
    return data





def return_final_out(data, prediction_days):
     #Now the data should be prepossessed to add the neccessary features, and remove the unneccessary ones
    data = preprocess(data)

    #Data should be splitted for the 2 models
    market_data = data.drop(['Close'], axis = 1)
    close_price_data = data[['Close']]


    #Scaling both data using already trained scalers
    market_data = market_data_scaler.transform(market_data)
    close_price_data = close_price_scaler.transform(close_price_data)



    market_features_pred = []

    for i in range (prediction_days):
        output = market_model.predict(market_data.reshape(1,market_data.shape[0],market_data.shape[1]))
        market_features_pred.append(output)

        market_data = market_data[1:] #removing the element at the 0th index and selcting the rest
        market_data = np.concatenate((market_data,output), axis = 0)

    market_features_pred = np.array(market_features_pred)


    final_output = []

    for i in range (prediction_days):
        output = close_price_model.predict(market_features_pred[i])
        final_output.append(output)


    final_output = np.array(final_output)


    #inverse scaling
    final_output = close_price_scaler.inverse_transform(final_output.reshape(-1,1))

    return final_output


def plot_graph(final_output,real_close_prices = pd.Series(None)):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(1,prediction_days+1)),
                             y=pd.Series(final_output.reshape(-1)).diff(), 
                             mode='lines+markers', 
                             name='Predicted Prices',
                             line=dict(color='royalblue', width=2),
                             marker=dict(size=10, color='red', symbol = 'circle')))
    
    

    if not real_close_prices.empty:
        fig.add_trace(go.Scatter(x = list(range(1,prediction_days+1)),
                             y= pd.Series(real_close_prices).diff(), 
                             mode='lines+markers', 
                             name='Actual Prices',
                             line=dict(color='green', width=2),
                             marker=dict(size=10, color='red', symbol = 'circle')))
    

    fig.update_layout(title="Stock price prediction for next 15 days",
                        xaxis_title="Time (Days)",   # Custom X-axis title
                        yaxis_title="Stock Price ",  # Custom Y-axis title

                        xaxis=dict(
                            showgrid=False, gridcolor="lightgray",  # Add X-axis grid
                            dtick=1,  # Show a tick every 10 points
                            tickformat="d",  # Integer format for X-axis
                            range = [0, prediction_days+1]
                        ),
                    
                        yaxis=dict(
                            showgrid=False, gridcolor="lightgray",  # Add Y-axis grid
                            range=[-100,100],  # Set Y-axis range manually
                            tickformat = '.2f',  # Force uniform tick spacing
                            dtick=50,  # Show a tick every 5 units
                        ),
                        
                        template="plotly_white",  # Change theme
                        margin=dict(l=40, r=40, t=40, b=40),
                        height = 600, width = 900,
                        # show_legend = True
                    )
        
    st.plotly_chart(fig)



# Predict Next 15 Days Price
if st.button("Predict"):

    data = get_stock_data(stock_name)  #this returns the past 60 days data

    final_output = return_final_out(data,prediction_days)
   
    plot_graph(final_output)

    

# Accuracy Check Feature
st.subheader("Check Model Accuracy")

end_date = st.date_input("End Date:", max_value=datetime.today().date())
start_date = (end_date - timedelta(days=60)).strftime('%Y-%m-%d')
if st.button("Show Predictions"):
    if end_date:

        data = get_stock_data(stock_name)  #this returns the past 60 days data
        data.index = data.index.tz_convert(None)
        real_close_prices = data.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]['Close']
        accuracy_output = return_final_out(data.loc[:pd.to_datetime(end_date)],prediction_days)
   
        plot_graph(final_output=accuracy_output,real_close_prices=real_close_prices)

        print(accuracy_output)
        print(real_close_prices)

    else:
        st.error("Please select a valid date range.")

