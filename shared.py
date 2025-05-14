from pathlib import Path

import numpy as np
import pandas as pd

app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "empty_stock.csv")

def get_stock_characteristics(stock : pd.DataFrame) -> pd.DataFrame:
    """
    Get the characteristics of the stock data.
    """
    stock['WAP'] = (stock['bid_price1'] * stock['bid_size1'] + \
                    stock['bid_price2'] * stock['bid_size2'] + \
                    stock['ask_price1'] * stock['ask_size1'] + \
                    stock['ask_price2'] * stock['ask_size2'])  \
    / (stock['bid_size1'] + stock['bid_size2'] + stock['ask_size1'] + stock['ask_size2'])


    # Add log returns calculation
    stock['log_return'] = np.log(stock['WAP'] / stock['WAP'].shift(1))

    window_size = 5
    stock['volatility'] = stock['log_return'].rolling(window=window_size).std()
    return stock

def get_stock_feature(stock : pd.DataFrame) -> pd.DataFrame:
    stock['ma5'] = stock['volatility'].rolling(window=5).mean()
    stock['ma10'] = stock['volatility'].rolling(window=10).mean()

    stock['future'] = stock['volatility'].shift(-1)
    return stock

def get_volume_feature(stock : pd.DataFrame) -> pd.DataFrame:
    stock['bs_ratio'] = \
        (stock['ask_size1'] + stock['ask_size2']) / \
        (stock['bid_size1'] + stock['bid_size2'])

    stock['bs_chg'] = stock['bs_ratio'].pct_change()

    stock['bd'] = stock['bid_size1'].pct_change() # Bid Delta
    stock['ad'] = stock['ask_size1'].pct_change() # Ask Delta


    # 1. OBV (On-Balance Volume)
    stock['OBV'] = np.where(stock['WAP'] > stock['WAP'].shift(1), 
                            stock['bid_size1'] + stock['ask_size1'], 
                            - (stock['bid_size1'] + stock['ask_size1'])).cumsum()

    # 2. VWAP (Volume Weighted Average Price)
    stock['VWAP'] = (stock['bid_price1'] * stock['bid_size1'] + stock['ask_price1'] * stock['ask_size1']) / \
                    (stock['bid_size1'] + stock['ask_size1'])

    # 3. Volume Moving Average (using bid and ask size as volume proxy)
    stock['Volume_MA'] = (stock['bid_size1'] + stock['ask_size1']).rolling(window=10).mean()

    return stock

def process_group(stock : pd.DataFrame) -> pd.DataFrame:
    """
    Process a group of stock data.
    """
    stock = get_stock_characteristics(stock)
    stock = get_stock_feature(stock)
    stock = get_volume_feature(stock)
    return stock