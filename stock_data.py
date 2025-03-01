#!/usr/bin/env python
# coding: utf-8

# # Dataset

# In[1]:


# %pip install yfinance


# In[2]:


#Importing libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[3]:


# stock = None


# def get_stock_data(ticker,period="max", interval="1d"):#the function retrieves the data of about the specified period with the interval mentioned - here 6month data with 1day interval
#     '''This function gets the name of stock(ticker name) from the user and creates a ticker object
#     and retrieves the basic historic stock price data from yahoo finance'''

#     global stock  #changing the stock variable that is outside the function


#     stock = yf.Ticker(ticker)     #creating a ticker object - simply an object that represents the specified stock.
#     data = stock.history(period=period, interval=interval, actions = True) #gets the historical price data from yahoo finance

#     return data

#checking the data from yahoo finance
# data = get_stock_data('AAPL')  #AAPL is the tickername for apple


# A ticker symbol is a unique series of letters assigned to a publicly traded company's stock or other securities for identification on a stock exchange.
# 
# For example:
# 
# AAPL → Apple Inc. (Nasdaq)
# 
# Ticker symbols vary in format depending on the exchange:
# 
# Usually the stock name is like this - symbolname.stock exchange name
# for-eg - 'Reliance.NS' where relianace is the symbol name and NS indicated national stock exchange of India.
# 
# Close is the closing price - the price of the stock when the stock market is closed on each day. Same goes for open. It is the proice of the stock when the stock market is opened.
# 
# High and low are the highest and lowest prices of the stock in each day

# In[4]:


def calculate_indicators(data):
    '''This function calculates differenct indices that efficiently represents the stock price movements
    and add it to the existing data'''

    #Trend Based Features (Momentum Indiactors) - Cptures the direction and strength of price movements
    data["SMA_50"] = data["Close"].rolling(window=50).mean()  # .rolling() method in Pandas is used to create a rolling window (or moving window) over a time series. This allows us to compute statistics like moving averages, standard deviations, and other aggregate functions over a fixed number of past values.
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["RSI"] = compute_rsi(data["Close"], 14)

    data["MACD"], data["Signal"] = compute_macd(data["Close"])
    data["Bollinger_Upper"], data['Bollinger_middle'],data["Bollinger_Lower"] = compute_bollinger_bands(data["Close"])
    data["VWAP"] = compute_vwap(data)
    
    #Volatility Based Features - Cpatures market uncertainity and price fluctuations
    data["ATR"] = compute_atr(data)
    data['Volatility_10'] = data['Close'].rolling(window=10).std()  #Rolling Standard deviation (Volatility)

    #Price Based features
    data['Daily_Return'] = data['Close'].pct_change()  # Measures the daily returns
    data['Cumulative_Return'] = (1+data['Daily_Return']).cumprod()  # 1 is added to convert % changes into a cumative growth factor

    #Volume Based Features - help detect market participation and smart money movement
    data['OBV'] = compute_OBV(data)  # On-Balance Volume
    data['VMA_10'] = data['Volume'].rolling(window = 10).mean()  #Calculate the moving avg for the volume data for every 10 days

    #Lagged Features - adds previous values to the data as features so that the model can easily learn patterns
    data['Close_Lag_1'] = data['Close'].shift(1) #adds previous day's close price
    data['Close_Lag_7'] = data['Close'].shift(7) # adds the previous weeks close price
    data['Close_Lag_30'] = data['Close'].shift(30) # adds the previous weeks close price
    
    data['Return_Lag_1'] = data['Daily_Return'].shift(1) #adds the previus days returns
    data['Return_Lag_7'] = data['Daily_Return'].shift(7) #adds the previus weeks returns
    data['Return_Lag_30'] = data['Daily_Return'].shift(30) #adds the previus month's returns
    
    #Advanced Statistical Features - capture deep insights from the price action
    data['Skewness'] = data['Daily_Return'].rolling(30).skew() #skewness
    data['Kurtosis'] = data['Daily_Return'].rolling(30).kurt() #calculates kurtosis - oth measures return distribution
    
    #Fibonacci based features - They are great for finding trend reversals, it predicts retracement and breakout points
    data['Fib_23.6'], data['Fib_38.2'], data['Fib_61.8'], data['Fib_78.6'] = compute_fibonacci(data)
    
    #Time Based Features - helps in identifying seasonality effects. It alos helps in finding the holiday effects and the recurring market behaviours
    data['Day'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['Is_Weekend'] = (data['Day']>=5).astype(int)

    #Features that indiacte smart money Flow - It captures big institutional moves
    # data['ADI'] = compute_ADI(data)  #returns Accumulation/Distribution Index
    data['Typical_Price'], data['Money_Flow'],data['MFI'] = compute_MFI(data) #returns money flow index
    

    
    
    return data


# **SMA 50**(50- Day Simple Moving Average) (Short-Term Trend) → Represents the average stock price over the last 50 days. It reflects short-term momentum and is used to gauge recent price trends.
# 
# **SMA_200 (200-Day Simple Moving Average)** – A trend-following indicator that calculates the average stock price over the past 200 days. It helps identify long-term trends and support/resistance levels. It helps determine the overall market direction and long-term sentiment.
# 
# **How do they affect the stock prices** :
# + When SMA 50 crosses above SMA 200, it signals strong upward momentum → BUY signal
# + When SMA 50 crosses below SMA 200, it signals downward momentum → SELL signal
# 
# 

# In[5]:


def compute_rsi(series, period=14):
    '''Computes the Relative strength index'''

    #calculate the difference of closing price of each day with the next one
    difference = series.diff(1)

    #calcluate the gain and loss -
    #a positive value in the diffence column indicates a gain
    #while the negative value indicates a loss
    gain = (difference.where(difference > 0, 0))
    loss = (abs(difference.where(difference < 0, 0))) #computes the absolute loss

    #calculate the gain and loss for a period of 14 days for each 14 days available consicutively
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss  #calculate rs
    rsi = 100 - (100 / (1 + rs))  #this is the formula for RSI
    return rsi


# **RSI (Relative Strength Index)** – A momentum indicator that measures the speed and change of price movements on a scale of 0 to 100. RSI above 70 indicates overbought conditions, while below 30 suggests oversold conditions.
# 
# + RSI doesn’t directly affect stock prices, but it influences trader behavior and market sentiment, leading to price movements.
# ​
# + If RSI rises above 70, traders may start selling (profit-taking).
# + If RSI drops below 30, traders may buy the dip, pushing prices up.
# 

# In[6]:


def compute_macd(series, short=12, long=26, signal=9):
    '''Computes the MACD'''
    short_ema = series.ewm(span=short, adjust=False).mean() #EMA of 12 days
    long_ema = series.ewm(span=long, adjust=False).mean() #EMA of 26 days

    macd = short_ema - long_ema #calculates the macd line
    signal_line = macd.ewm(span=signal, adjust=False).mean() #calculates the signal line
    return macd, signal_line


# **MACD (Moving Average Convergence Divergence)** – A trend-following momentum indicator that shows the relationship between two moving averages (typically the 12-day and 26-day EMA(exponential moving averages). It helps identify trend direction, strength, and potential reversals.
# 
# So how it works-
# First the exponential moving average (EMA) of 12 days and 26 days are calculated. EMA is similar to SMA but it assigns a greater weight to the  recent prices. Bith EMA will a series of values
# 
# The difference between the short and long EMA is called the macd line
# 
# The EMA of the macd line for a period of 9 days is called a signal line.
# 
# + When the MACD Line crosses above the Signal Line, it suggests increasing upward momentum.
#   + Traders buy the stock at this point.
# 
# + When the MACD Line crosses below the Signal Line, it suggests increasing downward momentum.
#   + Traders sell the stock at this point.

# In[7]:


def compute_bollinger_bands(series, window=20, num_std=2):
    '''Computes the Bollinger_bands'''

    sma = series.rolling(window=window).mean() #calculates the simple moving average(SMA) for 20 days
    std_dev = series.rolling(window=window).std() #std deviation of 20 days correspondimg to the sma

    upper_band = sma + (num_std * std_dev)
    middle_band = sma
    lower_band = sma - (num_std * std_dev)
    return upper_band, middle_band,lower_band


# **Bollinger_Bands** –Bollinger Bands measures volatility. The band expand and contarc based on the market volatility
# 
# Middle Band (SMA 20) → A 20-day Simple Moving Average (SMA)
# 
# Upper Band → SMA + (2 × Standard Deviation)
# 
# Lower Band → SMA - (2 × Standard Deviation)
# 
# **How Bollinger Bands Reflect Stock Prices?**
# + When Price Touches the Upper Band → Stock is overbought (possible reversal or correction).
# + When Price Touches the Lower Band → Stock is oversold (possible buying opportunity).
# + When Bands Expand → High volatility (price is making big moves).
# + When Bands Contract → Low volatility (price is in a consolidation phase).

# In[8]:


def compute_atr(data, period=14):
    '''Computes the ATR'''
    high_low = data["High"] - data["Low"] #The diffence between the high and low values of a stock
    high_close = np.abs(data["High"] - data["Close"].shift()) #The absolute difference between high and previous close value
    #.shift() is used to shift values of a series/dataframe up or down
    #this essentially helps to calculate the difference between the current high and preivious close.

    low_close = np.abs(data["Low"] - data["Close"].shift()) #the absolute difference between low value and the previous close value
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1) #Choosing the max of the above 3 values

    return tr.rolling(window=period).mean()  #the sma of the tr values


# **ATR (Average True Range)** – A volatility indicator that measures the range between a stock’s high and low prices over a set period. Higher ATR values indicate higher volatility.
# 
# It is a volatility indicator that measures how much a stock moves on average per day.
# 
# ATR is based on the True Range (TR), which captures the actual movement of a stock.
# For each day, the True Range (TR) is the largest of the following three values:
# 
# High - Low (Current day's range)
# High - Previous Close (Gap-up scenario)
# Low - Previous Close (Gap-down scenario)
# 
# The max of the above three is called the TR value.
# This ensures that ATR accounts for both intraday volatility and gaps between days.
# 
# Then a SMA or EMA is calculated over a specific period (usually 14) to get the average true range. Here we have done SMA for simplicity.

# In[9]:


def compute_vwap(data):
    '''Computes the Volume weighted average price'''
    price = (data["High"] + data["Low"] + data["Close"]) / 3 #taking the avg of the three - low high and close values
    vwap = (data['Volume']*price).cumsum() / data['Volume'].cumsum() #calculating the vwap

    return vwap


# **VWAP (Volume Weighted Average Price)** – A trading benchmark that gives the average price a stock has traded at throughout the day, weighted by volume. It helps traders determine the stock’s trend relative to its average price.
# 
# **How it is Calculated**
# vwap = sum(price*volume)/sum(volume)
# 
# here price = (high+low+close)/3
# sum is cumulative sum.

# In[10]:


def compute_OBV(data):
    '''Calculate the On Balance Volume'''
    price_difference = data['Close'].diff()  #calculates the price difference from the previous day
    price_difference = np.sign(price_difference) 
    #This gives 1 for positive change - indicating price increase
    #This gives -1 for negative change - indicating price decrease
    #This gives 0 for no change - indicating no price movement

    obv = (data['Volume']*price_difference).cumsum() #takes the the cumulative sum of the product of the volume and price difference
    
    return obv


# In[11]:


def compute_fibonacci(data,period = 30):
    '''Calculate multiple fibonacci retracement levels'''

    high = data['High'].rolling(period).max()
    low= data['Low'].rolling(period).min()

    data['Fib_23.6'] = low + (high - low) * 0.236
    data['Fib_38.2'] = low + (high - low) * 0.382
    data['Fib_61.8'] = low + (high - low) * 0.618
    data['Fib_78.6'] = low + (high - low) * 0.786

    return  data['Fib_23.6'], data['Fib_38.2'], data['Fib_61.8'], data['Fib_78.6']


# In[12]:


def compute_ADI(data):
    '''Calculates the Accumulation/Distribution Index'''
    data['ADI'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
    data['ADI'] = data['ADI'].cumsum()

    return  data['ADI']


def compute_MFI(data):
    '''Caluclates the Money Flow Index'''

    # Calculate the Typical Price
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3

    print(data['Typical_Price'].shape)
    print(data['Volume'].head())
    # Calculate the Money Flow
    data['Money_Flow'] = data['Typical_Price'] * data['Volume'].squeeze()
    print(data['Typical_Price'].head())

    # Identify Positive and Negative Money Flow
    data['Positive_MF'] = data['Money_Flow'].where(data['Typical_Price'] > data['Typical_Price'].shift(1), 0)
    data['Negative_MF'] = data['Money_Flow'].where(data['Typical_Price'] < data['Typical_Price'].shift(1), 0)

    # Sum Positive and Negative Money Flow over 14 periods
    positive_flow = data['Positive_MF'].rolling(14).sum()
    negative_flow = data['Negative_MF'].rolling(14).sum()

    # Calculate the Money Flow Index (MFI)
    money_flow_ratio = positive_flow / negative_flow
    data['MFI'] = 100 - (100 / (1 + money_flow_ratio))

    return data['Typical_Price'],  data['Money_Flow'],  data['MFI']




# In[13]:


def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "Dividend Yield": info.get("dividendYield"),
        "52 Week High": info.get("fiftyTwoWeekHigh"),
        "52 Week Low": info.get("fiftyTwoWeekLow"),
    }

#checking the function output
# get_fundamental_data('RELIANCE.NS')


# Market Capitalization (Market Cap) is the total value of a company's outstanding shares in the stock market. It represents the company's size and worth in the market.
# 
# Formula:  Stock Price × Shares, any change in stock price affects the market cap in real-time.
# If stock price increases, market cap rises.
# If stock price decreases, market cap falls.

# 
# 
# 1. **Market Cap (Market Capitalization)** – The total value of a company’s outstanding shares, calculated as **Market Cap = Share Price × Shares Outstanding**. It indicates the company’s size and overall market value.  
# 
# 2. **P/E Ratio (Price-to-Earnings Ratio)** – A valuation metric that compares a company’s stock price to its earnings per share (EPS). **P/E Ratio = Share Price / EPS**. A high P/E suggests the stock is overvalued, while a low P/E may indicate undervaluation.  
# 
# 3. **EPS (Earnings Per Share)** – A measure of a company’s profitability, calculated as **EPS = Net Income / Total Shares Outstanding**. It represents how much profit a company makes per share of stock.  
# 
# 4. **Dividend Yield** – The percentage of a company’s share price that is paid to shareholders as dividends annually. **Dividend Yield = (Annual Dividend per Share / Share Price) × 100**. A higher yield suggests better returns from dividends.  
# 
# 5. **52 Week High** – The highest price at which a stock has traded in the past 52 weeks. It helps investors understand the stock’s price range and volatility.  
# 
# 6. **52 Week Low** – The lowest price at which a stock has traded in the past 52 weeks. Investors use this to assess potential buying opportunities.  
# 
# 7. **Shares Outstanding** – The total number of a company’s shares that are currently held by investors, including institutional and retail investors. It affects market cap and EPS calculations.  

# In[14]:


def clean_data(data):
    data = data.dropna() #Some indices like- SMA, will have nan values for some of the first values
    data = data.drop(['Dividends','Stock Splits', 'High','Low','Open'], axis =1) #Remove these 2 columsn as sthey are un necessary for prediction
    return data



def preprocess(data):
    data = calculate_indicators(data)
    
    
    data = clean_data(data)

    return data.iloc[-1:-61:-1][::-1] #last 60 days data





# In[15]:


# nifty = [
#     "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "AMBUJACEM.NS",
#     "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
#     "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS",
#     "DIVISLAB.NS", "DRREDDY.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
#     "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
#     "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
#     "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
#     "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
#     "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
#     "UPL.NS", "WIPRO.NS"
# ]


# In[16]:


# nifty_200 = [
#     "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANITRANS.NS", "AMBUJACEM.NS",
#     "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
#     "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS",
#     "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS",
#     "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
#     "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
#     "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
#     "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
#     "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
#     "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
#     "UPL.NS", "WIPRO.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS",
#     "ACC.NS", "ADANIPOWER.NS", "ALKEM.NS", "AMARAJABAT.NS", "APOLLOHOSP.NS",
#     "APOLLOTYRE.NS", "ASHOKLEY.NS", "AUROPHARMA.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
#     "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHEL.NS",
#     "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS", "CASTROLIND.NS", "CHOLAFIN.NS",
#     "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS",
#     "DLF.NS", "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
#     "GAIL.NS", "GLAND.NS", "GLENMARK.NS", "GMRINFRA.NS", "GODREJCP.NS",
#     "GODREJPROP.NS", "HAVELLS.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HONAUT.NS",
#     "IBULHSGFIN.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS",
#     "INDIGO.NS", "INDUSTOWER.NS", "IPCALAB.NS", "IRCTC.NS", "JINDALSTEL.NS",
#     "JUBLFOOD.NS", "L&TFH.NS", "LICHSGFIN.NS", "LUPIN.NS", "M&MFIN.NS",
#     "MANAPPURAM.NS", "MFSL.NS", "MGL.NS", "MINDTREE.NS", "MOTHERSUMI.NS",
#     "MPHASIS.NS", "MRF.NS", "NAM-INDIA.NS", "NAUKRI.NS", "NAVINFLUOR.NS",
#     "NMDC.NS", "NOCIL.NS", "OBEROIRLTY.NS", "OFSS.NS", "PAGEIND.NS",
#     "PEL.NS", "PETRONET.NS", "PFIZER.NS", "PIIND.NS", "PNB.NS",
#     "POLYCAB.NS", "PVR.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS",
#     "SAIL.NS", "SBICARD.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS",
#     "SRTRANSFIN.NS", "STAR.NS", "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS",
#     "TATACOMM.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS",
#     "TVSMOTOR.NS", "UBL.NS", "VEDL.NS", "VOLTAS.NS", "WHIRLPOOL.NS",
#     "ZEEL.NS", "ZYDUSLIFE.NS"
# ]


# In[17]:


# for ticker in nifty:
#     try:
#         data = main(ticker)
#     except:
#         continue

#     if data.shape != 0:
#         data.to_csv(f'stock_50\{ticker}.csv')
    



# In[18]:


# data.to_csv('NESTLEIND.NS.csv')


# In[19]:


# data.shape


# In[20]:


# data.head()


# In[ ]:




