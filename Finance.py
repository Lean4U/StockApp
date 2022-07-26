#!/usr/bin/env python
# coding: utf-8

# In[9]:


import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App

## Shown are the stock closing price and volume of Amazon!

""")

# Input text widgit
stock = st.radio("The current stock ticker is:", ('AMZN', 'AAL', 'BAC'))


# http://towardsdatascience.com/how-to-get-stick-data-using-python-c0de1df17e75
# define the ticker symbol

tickerSymbol=stock

# get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# get the historical proces for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

# Open High Low Close Volume Dividents Stock Splits

st.write("""
## Stock Price
""")

st.line_chart(tickerDf.Close)

st.write("""
## Stock Volume
""")

st.line_chart(tickerDf.Volume)



# In[ ]:
