import pandas as pd

def feature_engineering(dataframe, rsi_per = 7, bb_per = 20) :
    data = dataframe.copy()
    date_to_exclude = pd.Timestamp("today").normalize()
    data = data[data.index!=date_to_exclude]
    # Calcul des indicateurs
    # MA
    data["SMA7"] = data["close"].rolling(window = 7, min_periods = 1).mean()
    data["SMA14"] = data["close"].rolling(window = 14, min_periods = 1).mean()
    
    # RSI
    delta = data["close"].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0 # On ne garde que les hausses
    loss[loss > 0] = 0 # On ne garde que les baisses (en négatif)
    
    avg_gain = gain.rolling(window=rsi_per, min_periods = 1).mean()
    avg_loss = abs(loss.rolling(window=rsi_per, min_periods = 1).mean())
    
    data[f"RSI{rsi_per}"] = avg_gain/(avg_gain+avg_loss)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = data['close'].ewm(span=12, adjust=False, min_periods = 1).mean() # Adjust for recursivity
    ema_26 = data['close'].ewm(span=26, adjust=False,min_periods = 1).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False, min_periods = 1).mean()
    data['Histogram'] = data['MACD'] - data['Signal_Line']
    
    # Bollinger Bands
    data["BB_Midle"] = data['close'].rolling(window=bb_per, min_periods = 1).mean()
    bb_rolling_std =  data['close'].rolling(window=bb_per, min_periods = 1).std()
    data["BB_Upper"] = data["BB_Midle"]+(2*bb_rolling_std)
    data["BB_Lower"] = data["BB_Midle"]-(2*bb_rolling_std)
    
    # Let's compute the target
    data["lag1_close"] = data["close"].shift(1)
    data["lag1_high"] = data["high"].shift(1)
    data["lag1_low"] = data["low"].shift(1)
    
    return data