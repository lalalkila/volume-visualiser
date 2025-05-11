from faicons import icon_svg
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from xgboost import XGBRegressor, XGBClassifier  # or XGBClassifier for classification
from sklearn.model_selection import train_test_split

# Import data from shared.py
from shared import app_dir, df, get_stock_characteristics, get_stock_feature, get_volume_feature, process_group

from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  

data = reactive.Value(df)
bucket_size = 30
features = [
    'volatility',
    'ma5',
    'ma10',
    'future',
    'bs_ratio',
    'bs_chg',
    'bd',
    'ad',
    'OBV',
    'VWAP',
    'Volume_MA',
    'Lagged_Volume'
]

VOL_MODEL_FEATURES = ['volatility', 'ma5', 'bs_ratio', 'bs_chg', 'bd', 'ad',  'OBV', 'VWAP', 'Volume_MA', 'Lagged_Volume']

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload a CSV file", accept=".csv"),
        ui.input_select(
            "timeid",
            "Time ID",
            [],  # Start with empty choices, to be updated later
            selected=None,
        ),
        ui.input_selectize(  
            "display_features",  
            "Select features below:",  
            {feature : feature for feature in features},
            selected=['volatility'],  
            multiple=True,  
        ),  
    ),
    ui.layout_column_wrap(
        ui.value_box(
            "Number of timeIDs",
            ui.output_text("count"),
            showcase=icon_svg("earlybirds"),
        ),
        ui.value_box( # Added a second value box, but it seems to do the same thing.
            "Base Model RMSE",
            ui.output_text("count1"),
            showcase=icon_svg("earlybirds"),
        ),
        ui.value_box( # Added a third value box, but it seems to do the same thing.
            "Volume Model RMSE",
            ui.output_text("count3"),
            showcase=icon_svg("earlybirds"),
        ),
        fill=False,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Features explorer"),
            output_widget("prediction"),
            output_widget("plot"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Data explorer"),
            ui.output_data_frame("summary_features"),
            full_screen=True,
        ),
    ),
    ui.include_css(app_dir / "styles.css"),
    title="Volume predictions",
    fillable=True,
)


def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.file)
    def _():
        if input.file():
            file_info = input.file()[0]
            try:
                df = pd.read_csv(file_info["datapath"])
                df['bucket'] = np.floor(df['seconds_in_bucket'] / 30)
                df = df.groupby(['time_id', 'bucket']).mean()[['bid_price1','ask_price1','bid_price2','ask_price2','bid_size1','ask_size1','bid_size2', 'ask_size2']].round(4).reset_index()
                data.set(df)
                # print(data.get().head()) #  Good for debugging, but remove in production
                ui.update_select(
                    "timeid",
                    label="Choose TimeID:",
                    choices=df["time_id"].unique().tolist(),
                    selected=str(df["time_id"].unique()[0]) if not df.empty else None, #handle empty df
                )
            except Exception as e:
                print(f"Error reading CSV: {e}") # Important: Handle errors!
                #  Consider showing a user-friendly message in the UI, not just printing.

    @reactive.calc
    def filtered_df():
        df = data.get()
        if input.timeid():
            stock =  df[df["time_id"] == int(input.timeid())]
            return stock
        else:
            return df
    
    @reactive.calc
    def stock_features():
        stock = filtered_df()
        stock = get_stock_characteristics(stock)
        stock = get_stock_feature(stock)
        stock = get_volume_feature(stock)
        return stock
    
    @reactive.calc
    def model_data():
        if data.get().empty:
            return pd.DataFrame()
        stock = data.get()
        stock = get_stock_characteristics(stock)
        stock = get_stock_feature(stock)
        stock = get_volume_feature(stock)
        print(stock.head()) # Good for debugging, but remove in production
        stock = stock.groupby('time_id', group_keys=False).apply(process_group)
        stock = stock.dropna()
        print('got model data')
        return stock
    
    @reactive.calc
    def base_model():
        if data.get().empty:
            return pd.DataFrame()
        stock = model_data()
        X = stock[['volatility', 'ma5']]
        y = stock['future']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        model = XGBRegressor() 
        model.fit(X_train, y_train)
        print('got base model')
        return model
    
    @reactive.calc
    def get_residual():
        if data.get().empty:
            return pd.DataFrame()
        stock = model_data()
        stock['residual'] = stock['future'] - base_model().predict(stock[['volatility', 'ma5']])
        print('got residual')
        return stock
    
    @reactive.calc
    def vol_model():
        if data.get().empty:
            return pd.DataFrame()
        stock = get_residual()
        X = stock[VOL_MODEL_FEATURES]
        y = stock['residual']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        model = XGBRegressor() 
        model.fit(X_train, y_train)
        return model
    
    @reactive.calc
    def get_vol_residual():
        if data.get().empty:
            return pd.DataFrame()
        stock = get_residual()
        stock['vol_residual'] = stock['residual'] - vol_model().predict(stock[VOL_MODEL_FEATURES])
        print('got vol residual')
        print(stock.head())
        return stock
    
    @render.data_frame
    def summary_features():
        if stock_features().empty:
            return pd.DataFrame()
        return stock_features()[['time_id', 'bucket', 'WAP', 'log_return', 'volatility', 'ma5', 'ma10', 'future', 'bs_ratio', 'bs_chg', 'bd', 'ad', 'OBV', 'VWAP', 'Volume_MA', 'Lagged_Volume']].round(4)

    @render_widget  
    def prediction():  
        if get_vol_residual().empty or filtered_df().empty or input.timeid() is None:
            return None
        
        stock = get_vol_residual()[get_vol_residual()["time_id"] == int(input.timeid())]

        fig = go.Figure()
        fig.add_trace(
                go.Scatter(
                    x=filtered_df()[filtered_df()['future'].notna()]["bucket"],
                    y=filtered_df()[filtered_df()['future'].notna()]['volatility'],
                    mode="lines",
                    name='volatility',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['volatility'] - stock['residual'],
                    mode="lines",
                    name='Base Model',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['volatility'] - stock['residual'] - stock['vol_residual'],
                    mode="lines",
                    name='Volume Model',
                )
            )
        return fig
    
    @render_widget  
    def plot():  
        if stock_features().empty:
            return None
        print(stock_features().head()) # Good for debugging, but remove in production
        fig = go.Figure()
        features = input.display_features()
        for feature in features:
            fig.add_trace(
                go.Scatter(
                    x=stock_features()["bucket"],
                    y=stock_features()[feature],
                    mode="lines",
                    name=feature,
                )
            )
        return fig

    @render.text
    def count():
        # Depend on data.  This is the key change.
        df = data.get()
        return str(df["time_id"].unique().shape[0])

    @render.text
    def count1(): #Fixes count2 to be unique
        # print(input.timeid())
        # get_vol_residual()
        if stock_features().empty:
            return 0
        return np.sqrt(np.mean(np.square(get_residual()['residual'])))

    @render.text
    def count3():
        if stock_features().empty:
            return 0
        return np.sqrt(np.mean(np.square(get_vol_residual()['vol_residual'])))

app = App(app_ui, server)