from faicons import icon_svg
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from xgboost import XGBRegressor, XGBClassifier  # or XGBClassifier for classification
from sklearn.model_selection import train_test_split

# Import data from shared.py
from shared import app_dir, df, get_stock_characteristics, get_stock_feature, get_volume_feature, process_group

from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  

data = reactive.Value(df)
bucket_size = 30
features = [
    'ma5',
    'ma10',
    'bs_ratio',
    'bs_chg',
    'bd',
    'ad',
    'OBV',
    'VWAP',
    'Volume_MA',
]

VOL_MODEL_FEATURES = ['volatility', 'ma5', 'bs_ratio', 'bs_chg', 'bd', 'ad',  'OBV', 'VWAP', 'Volume_MA']

app_ui = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel(
        "App Introduction",
        ui.layout_columns(
            ui.output_ui("data_intro")
        )
    ),


    # Second tab - Main dashboard
    ui.nav_panel(
        "Interactive Model",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_file("file", "Model Explorer\nUpload a CSV file", accept=".csv"),
                ui.input_select(
                    "timeid",
                    "Time ID",
                    [],  # Start with empty choices, to be updated later
                    selected=None,
                ),
                ui.input_selectize(  
                    "display_features",  
                    "Select features below (Max 4):",  
                    {feature : feature for feature in features},
                    selected=['ad', 'bd', 'OBV', 'Volume_MA'],  
                    multiple=True,  
                    options={'maxItems': 4},
                ),  
            ),
            ui.layout_column_wrap(
                ui.value_box(
                    "Number of timeIDs",
                    ui.output_text("count"),
                    showcase=icon_svg("calendar"),
                ),
                ui.value_box( # Added a second value box, but it seems to do the same thing.
                    "Base Model RMSE",
                    ui.output_text("count1"),
                    showcase=icon_svg("robot"),
                ),
                ui.value_box( # Added a third value box, but it seems to do the same thing.
                    "Volume Model RMSE",
                    ui.output_text("count3"),
                    showcase=icon_svg("cube"),
                ),
                fill=False,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Model explorer"),
                    output_widget("prediction"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Feature explorer"),
                    output_widget("feature_plots"),
                    full_screen=True,
                ),
                # ui.card(
                #     ui.card_header("Data explorer"),
                #     ui.output_data_frame("summary_features"),
                #     full_screen=True,
                # ),
            ),
            fillable=True,
        ),
    ),
    id="navbar",
    header=ui.include_css(app_dir / "styles.css"),
    title="Volume predictions",
    window_title="Volume predictions",
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
                ui.update_select(
                    "timeid",
                    label="Choose TimeID:",
                    choices=df["time_id"].unique().tolist(),
                    selected=str(df["time_id"].unique()[0]) if not df.empty else None, #handle empty df
                )
            except Exception as e:
                print(f"Error reading CSV: {e}") # Important: Handle errors!

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
        stock = stock.groupby('time_id', group_keys=False).apply(process_group)
        stock_with_na = stock.copy()
        stock = stock.dropna()
        return stock
    
    @reactive.calc
    def base_model():
        if data.get().empty:
            return pd.DataFrame()
        stock = model_data()
        X = stock[['volatility', 'ma5']]
        y = stock['future']
        stock = stock.sort_values(["time_id", "bucket"])
        split_index = int(len(stock) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        model = XGBRegressor() 
        model.fit(X_train, y_train)
        return model
    
    @reactive.calc
    def get_residual():
        if data.get().empty:
            return pd.DataFrame()
        stock = model_data()
        stock['base_pred'] = base_model().predict(stock[['volatility', 'ma5']])
        stock['residual'] = stock['future'] - stock['base_pred']
        return stock
    
    @reactive.calc
    def vol_model():
        if data.get().empty:
            return pd.DataFrame()
        stock = get_residual()
        X = stock[VOL_MODEL_FEATURES]
        y = stock['residual']
        stock = stock.sort_values(["time_id", "bucket"])
        split_index = int(len(stock) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        model = XGBRegressor() 
        model.fit(X_train, y_train)
        return model
    
    @reactive.calc
    def get_vol_residual():
        if data.get().empty:
            return pd.DataFrame()
        stock = get_residual()
        stock['vol_residual'] = vol_model().predict(stock[VOL_MODEL_FEATURES])
        return stock
    
    @render.data_frame
    def summary_features():
        if stock_features().empty:
            return pd.DataFrame()
        return stock_features()[['time_id', 'bucket', 'WAP', 'log_return', 'volatility', 'ma5', 'ma10', 'future', 'bs_ratio', 'bs_chg', 'bd', 'ad', 'OBV', 'VWAP', 'Volume_MA']].round(4)

    @render_widget  
    def feature_plots():  
        if stock_features().empty:
            return None
       
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=input.display_features()
        )
       
        features = input.display_features()
        for i, feature in enumerate(features):
            row = i // 2 + 1
            col = i % 2 + 1
           
            fig.add_trace(
                go.Scatter(
                    x=stock_features()["bucket"],
                    y=stock_features()[feature],
                    mode="lines",
                    name=feature
                ),
                row=row, col=col
            )
       
        fig.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
        )


        return fig
        
    @render_widget  
    def prediction():  
        if get_vol_residual().empty or filtered_df().empty or input.timeid() is None:
            return None
        
        stock = get_vol_residual()[get_vol_residual()["time_id"] == int(input.timeid())]

        fig = go.Figure()
        fig.add_trace(
                go.Scatter(
                    x=filtered_df()[filtered_df()['future'].notna()]["bucket"],
                    y=filtered_df()[filtered_df()['future'].notna()]['future'],
                    mode="lines",
                    name='volatility',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['base_pred'],
                    mode="lines",
                    name='Base Model',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['base_pred'] + stock['vol_residual'],
                    mode="lines",
                    name='Volume Model',
                )
            )
        
        fig.update_layout(
            legend=dict(
                x=1.02,        # Slightly outside the main plot (right side)
                y=0.5,         # Middle vertically
                xanchor='left',
                yanchor='middle'
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
        if stock_features().empty:
            return 0
        return np.sqrt(np.mean(np.square(get_residual()['residual'])))

    @render.text
    def count3():
        if stock_features().empty:
            return 0
        return np.sqrt(np.mean(np.square(get_vol_residual()['vol_residual'])))
    
    @render.ui
    def data_intro():

        md = ui.markdown(
            """
            # 📈 Volatility Prediction Shiny App  
  

            ### Adjusted Residual Model with Volume Features

            Welcome to our Shiny app for predicting stock volatility using an **Adjusted Residual Model** that leverages key **volume-driven features**. This tool is designed to assist traders, analysts, and researchers in forecasting short-term price fluctuations by combining traditional volatility modeling with volume-based signals.

            ---

            ## 🔍 What Is the Adjusted Residual Model?

            The **Adjusted Residual Model** enhances baseline volatility predictions by adjusting their residuals using XGBoost algorithms. These adjustments are informed by features derived from trading volume, allowing for more responsive and accurate volatility forecasts, particularly during periods of unusual market activity.

            ---

            ## 🧠 Features Used in the Model

            The model uses a carefully engineered set of volume and price-based features, collectively defined as:

            Our model uses the following engineered features, which capture various dimensions of market activity and price-volume interaction:

            - ma5: 5-period moving average of weighted average price.

            - bs_ratio: Bid-ask size ratio, indicating supply-demand imbalance.

            - bs_chg: Change in bid-ask spread, capturing short-term liquidity shifts.

            - bd: Buy-side depth—aggregate buy-side volume near the best bid.

            - ad: Ask-side depth—aggregate sell-side volume near the best ask.

            - OBV (On-Balance Volume): A cumulative volume-based momentum indicator.

            - VWAP (Volume-Weighted Average Price): The average price weighted by volume.

            - Volume_MA: Moving average of trade volume, capturing trends in trading intensity.

            ## ⚙️ App Functionality

            - Visualisation: Interactive plots for volatility predictions.

            - Model Diagnostics: Residual analysis and performance metrics.

            - Customization: Choose different dataset, time id and features to predict and visualise

            """,
        )

        return ui.div(md, class_="my-3 lead")

app = App(app_ui, server)