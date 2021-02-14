"""
    Copyright (c) 2021 Brad Duy
    Unauthorized copying of this file, via any medium is strictly prohibited
"""

import streamlit as st 
from datetime import date 
import yfinance as yf 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go 

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title("Duy's stock prediction app")

stocks = ("BTC-USD", "TSLA", "AAPL", "GOOG", "GME")
selectedStock = st.selectbox("Select dataset for prediction", stocks)

numberYears = st.slider("Years of prediction: ", 1, 4)
period = numberYears * 365

@st.cache
def loadData(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

dataLoadState = st.text("Load data...")
data = loadData(selectedStock)
dataLoadState.text("Loading data... done")

st.subheader("Raw dataframe")
st.write(data.tail())

def plotRawData():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plotRawData()

# forecasting
dfTrain = data[['Date', 'Close']]
dfTrain = dfTrain.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(dfTrain)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecast dataframe")
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.write('Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)


