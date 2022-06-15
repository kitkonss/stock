import streamlit as sl
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

sl.title("Stock Prediction")

stocks = ("PTT.BK","AOT.BK")
selected_stocks = sl.selectbox("Select dataset for prediction", stocks)

n_years = sl.slider("Years of prediction:", 1, 4)
period = n_years*365

@sl.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = sl.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Done")

sl.subheader("Raw data")
sl.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    sl.plotly_chart(fig)
    
plot_raw_data()

#ทำนาย
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

sl.subheader("Forecast data")
sl.write(data.tail())

sl.write("forecast data")
fig1=plot_plotly(m, forecast)
sl.plotly_chart(fig1)

sl.write("forecast components")
fig2 = m.plot_components(forecast)
sl.write(fig2)