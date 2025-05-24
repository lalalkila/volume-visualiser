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
    stock['ma3'] = stock['volatility'].rolling(window=3).mean()
    stock['ma5'] = stock['volatility'].rolling(window=5).mean()
    # stock['ma10'] = stock['volatility'].rolling(window=10).mean()
    stock['mid_price'] = (stock['bid_price1'] + stock['ask_price1']) / 2
    stock['spread_lvl_1'] = stock['bid_price1'] - stock['ask_price1']
    stock['spread_lvl_2'] = stock['bid_price2'] - stock['ask_price2']


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
    stock['Volume_MA'] = (stock['bid_size1'] + stock['ask_size1']).rolling(window=5).mean()

    stock['volume_momentum_3'] = stock['Volume_MA'].pct_change(3)
    stock['volume_momentum_5'] = stock['Volume_MA'].pct_change(5)
    stock['volume_trend'] = stock['Volume_MA'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )
    
    # Volume-price relationship
    stock['volume_price_corr'] = stock['Volume_MA'].rolling(5).corr(stock['mid_price'])
    stock['vwap_deviation'] = (stock['mid_price'] - 
                                      (stock['mid_price'] * stock['Volume_MA']).rolling(5).sum() / 
                                      stock['Volume_MA'].rolling(5).sum())
    
    # Enhanced order flow features
    stock['order_flow_imbalance'] = (stock['bd'] - stock['ad']) / (stock['bd'] + stock['ad'] + 1e-8)
    stock['cumulative_order_flow'] = stock['order_flow_imbalance'].rolling(5).sum()
    
    # Volume volatility and regime changes
    stock['volume_volatility'] = stock['Volume_MA'].rolling(5).std()
    stock['volume_regime'] = (stock['Volume_MA'] > stock['Volume_MA'].rolling(5).quantile(0.75)).astype(int)
    
    # Bid-ask spread dynamics
    stock['bs_volatility'] = stock['bs_ratio'].rolling(5).std()
    stock['bs_momentum'] = stock['bs_chg'].rolling(3).mean()
    
    # Volume-based mean reversion signals
    stock['volume_zscore'] = (stock['Volume_MA'] - stock['Volume_MA'].rolling(5).mean()) / stock['Volume_MA'].rolling(5).std()
    stock['volume_percentile'] = stock['Volume_MA'].rolling(5).rank(pct=True)
    
    # Interaction terms (volume × price features)
    stock['volume_ma_interaction'] = stock['Volume_MA'] * stock['ma5']
    stock['bs_volume_interaction'] = stock['bs_ratio'] * stock['Volume_MA']

    return stock

def process_group(stock : pd.DataFrame) -> pd.DataFrame:
    """
    Process a group of stock data.
    """
    stock = get_stock_characteristics(stock)
    stock = get_stock_feature(stock)
    stock = get_volume_feature(stock)
    return stock