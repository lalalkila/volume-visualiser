import time

from faicons import icon_svg
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import data from shared.py
from shared import app_dir, df, get_stock_characteristics, get_stock_feature, get_volume_feature, process_group

from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget  

data = reactive.Value(df)
base_runtime = reactive.Value(0)
vol_runtime = reactive.Value(0)
bucket_size = 30
time_group_size = 20
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
# BASE_MODEL_FEATURES = ['past','ma5', 'mid_price', 'spread_lvl_1', 'spread_lvl_2']
# VOL_MODEL_FEATURES = ['past','bs_ratio', 'bs_chg', 'bd', 'ad',  'OBV', 'VWAP', 'Volume_MA']

BASE_MODEL_FEATURES = ['ma5', 'mid_price','volatility']
VOL_MODEL_FEATURES = ['base_pred', 'volatility',
                    'volume_momentum_5', 'volume_trend',
                    'volume_price_corr', 'vwap_deviation', 'order_flow_imbalance',
                    'cumulative_order_flow', 'volume_volatility', 
                    'bs_volatility', 'bs_momentum',
                    'volume_ma_interaction', 'bs_volume_interaction'
                    ]

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
                ui.input_file("file", "Upload a CSV file", accept=".csv"),
                ui.input_select(
                    "timeid",
                    "Time ID",
                    [],  # Start with empty choices, to be updated later
                    selected=None,
                ),
                ui.input_select(
                    "stockid",
                    "Stock ID",
                    [],  # Start with empty choices, to be updated later
                    selected=None,
                ),
            ),
            ui.layout_column_wrap(
                ui.output_ui("traintime"),
                ui.output_ui("directionalAccuracy"),
                ui.output_ui("improvement"),
                fill=False,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Volatility Forecasting"),
                    ui.output_ui("model_explorer_content"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Feature Importance"),
                    output_widget("feature_plots"),
                    full_screen=True,
                ),
                col_widths=(8, 4),
            ),
            fillable=True,
        ),
    ),
    id="navbar",
    header=ui.include_css(app_dir / "styles.css"),
    title="Volume-Volatility Eval",
    window_title="Volume-Volatilty Eval",
    fillable=True,
    
)


def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.file)
    def _():
        if input.file():
            print("hello")
            file_info = input.file()[0]
            try:
                print(file_info["datapath"])
                df = pd.read_csv(file_info["datapath"], delimiter='\t')
                df['bucket'] = np.floor(df['seconds_in_bucket']/ bucket_size)
                df = df.groupby(['stock_id', 'time_id', 'bucket']).mean()[['bid_price1','ask_price1','bid_price2','ask_price2','bid_size1','ask_size1','bid_size2', 'ask_size2']].round(4).reset_index()

                # Get all unique stock_ids in the dataframe
                unique_stock_ids = df['stock_id'].unique()

                # Create a template with all combinations of stock_id and time_id
                min_time_id = df['time_id'].min()
                max_time_id = df['time_id'].max()
                time_range = range(min_time_id, max_time_id + 1)

                # Create a cross join between stock_ids and time_ids
                template = pd.DataFrame([(stock_id, time_id) 
                                    for stock_id in unique_stock_ids 
                                    for time_id in time_range],
                                    columns=['stock_id', 'time_id'])

                # Sort the result
                # df_complete = df_complete.sort_values(by=['stock_id', 'time_id', 'bucket'])
                # split_index = int(len(df_complete) * 0.0125)
                # df_complete = df_complete.iloc[:split_index]
                # y = y.iloc[:split_index]

                df['time_id_group'] = np.floor(df['time_id'] / time_group_size)

                data.set(df)
                
                ui.update_select(
                    "timeid",
                    label="Choose TimeID:",
                    choices=df["time_id"].unique().tolist(),
                    selected=str(df["time_id"].unique()[0]) if not df.empty else None, #handle empty df
                )
                ui.update_select(
                    "stockid",
                    label="Choose StockID:",
                    choices=df["stock_id"].unique().tolist(),
                    selected=str(df["stock_id"].unique()[0]) if not df.empty else None, #handle empty df
                )
            except Exception as e:
                print(f"Error reading CSV: {e}") # Important: Handle errors!

    @reactive.calc  
    def stock_df():
        df = data.get()
        if input.timeid() and input.stockid():
            stock = df[df["stock_id"] == int(input.stockid())]
            return stock
        else:
            return pd.DataFrame(columns=df.columns)

    @reactive.calc  
    def filtered_df():
        df = stock_df()
        if input.timeid() and input.stockid():
            stock = df[df["time_id"] == int(input.timeid())]
            return stock
        else:
            return pd.DataFrame(columns=df.columns)
    
    @reactive.calc
    def stock_features():
        stock = filtered_df()
        stock = get_stock_characteristics(stock)
        stock = get_stock_feature(stock)
        stock = get_volume_feature(stock)
        return stock
    
    @reactive.calc
    def model_data():
        stock = stock_df()
        if stock.empty or input.timeid() is None:
            return pd.DataFrame(columns=df.columns)
        stock = stock[stock['time_id_group'] == np.floor(int(input.timeid()) / time_group_size)]
        stock = get_stock_characteristics(stock)
        stock = get_stock_feature(stock)
        stock = get_volume_feature(stock)
        stock = stock.groupby('time_id', group_keys=False).apply(process_group)

        stock = stock.dropna()

        return stock
    
    @reactive.calc
    def model_data_no_dropna():
        stock = stock_df()
        if stock.empty or input.timeid() is None:
            return pd.DataFrame(columns=df.columns)
        stock = stock[stock['time_id_group'] == np.floor(int(input.timeid()) / time_group_size)]
        stock = get_stock_characteristics(stock)
        stock = get_stock_feature(stock)
        stock = get_volume_feature(stock)
        stock = stock.groupby('time_id', group_keys=False).apply(process_group)

        return stock
    
    @reactive.calc
    def base_model():
        stock = model_data()
        if stock.empty:
            return None
        X = stock[BASE_MODEL_FEATURES]
        y = stock['future']
        stock = stock.sort_values(["time_id", "bucket"])
        # split_index = int(len(stock) * 0.8)
        # X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        # y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        y_train_scaled = y_train * 10000
        start = time.time()
        model = XGBRegressor(
                                    n_estimators=200,
                                    max_depth=4,
                                    learning_rate=0.05,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_alpha=0.1,
                                    reg_lambda=1.0,
                                    random_state=42,
                                    verbosity=0,
                                    objective='reg:squarederror'
                                )
        # model = LinearRegression()
        model.fit(X_train, y_train_scaled)
        end = time.time()
        base_runtime.set(end - start)

        return model
    
    @reactive.calc
    def get_residual():
        stock = model_data()
        if stock.empty:
            return pd.DataFrame(columns=df.columns)
        stock['base_pred'] = base_model().predict(stock[BASE_MODEL_FEATURES]) / 10000
        stock['residual'] = stock['future'] - stock['base_pred']

        pred = np.clip(stock['base_pred'], 1e-8, None)
        true = np.clip(stock['future'], 1e-8, None)
        print(f'QLIKE BASE {np.mean(np.log(pred) + true / pred)}')

        return stock
    
    @reactive.calc
    def vol_model():
        stock = get_residual()
        if stock.empty:
            return None
        X = stock[VOL_MODEL_FEATURES]
        y = stock['residual']
        stock = stock.sort_values(["time_id", "bucket"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        y_train_scaled = y_train * 10000
        start = time.time()
        model = XGBRegressor(
                                    n_estimators=200,
                                    max_depth=4,
                                    learning_rate=0.05,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_alpha=0.1,
                                    reg_lambda=1.0,
                                    random_state=42,
                                    verbosity=0,
                                    objective='reg:squarederror'
                                )
        model.fit(X_train, y_train_scaled)
        end = time.time()
        vol_runtime.set(end - start)

        return model
    
    @reactive.calc
    def get_vol_residual():
        stock = get_residual()
        if stock.empty or vol_model() is None:
            return pd.DataFrame(columns=df.columns)
        stock['vol_pred'] = vol_model().predict(stock[VOL_MODEL_FEATURES]) / 10000
        stock['vol_residual'] = stock['future'] - (stock['base_pred'] + stock['vol_pred'])

        pred = np.clip(stock['vol_residual'], 1e-8, None)
        true = np.clip(stock['future'], 1e-8, None)
        print(f'QLIKE VOL {np.mean(np.log(pred) + true / pred)}')

        return stock
    
    @render_widget
    def feature_plots():
        if base_model() is None or vol_model() is None:
            return None
        
        base = base_model()
        vol = vol_model()

        if not hasattr(base, "feature_importances_") or not hasattr(vol, "feature_importances_"):
            return None

        # Get feature importances from both models
        base_importances = base.feature_importances_
        base_features = BASE_MODEL_FEATURES
        
        vol_importances = vol.feature_importances_
        vol_features = VOL_MODEL_FEATURES
        print(vol_importances)

        # Create subplot with 2 rows
        fig = sp.make_subplots(
            rows=2, cols=1,
            # subplot_titles=("Base Model Feature Importance", "Volume Model Feature Importance")
        )

        # Volume model bar chart
        fig.add_trace(
            go.Bar(
                x=vol_importances,
                y=vol_features,
                orientation='h',
                marker=dict(color=px.colors.qualitative.Plotly[1]),
                name="Volume Model"
            ),
            row=2, col=1
        )

        # Base model bar chart
        fig.add_trace(
            go.Bar(
                x=base_importances,
                y=base_features,
                orientation='h',
                marker=dict(color=px.colors.qualitative.Plotly[2]),
                name="Base Model"
            ),
            row=1, col=1
        )

        fig.update_layout(
            annotations=[
                    dict(
                        text="Base Model Feature Importance",
                        xref="paper", yref="paper",
                        x=0, y=1.07,  # left aligned (near 0), y=top of first subplot
                        showarrow=False,
                        font=dict(size=14),
                        align="left"
                    ),
                    dict(
                        text="Volume Model Feature Importance",
                        xref="paper", yref="paper",
                        x=0, y=0.46,  # left aligned, y=middle lower for second subplot
                        showarrow=False,
                        font=dict(size=14),
                        align="left"
                    )
                ],
                margin=dict(l=55, r=20, t=40, b=20),
                showlegend=False,
        )

        fig.update_yaxes(tickfont=dict(size=8), automargin=True, row=1, col=1)
        fig.update_yaxes(tickfont=dict(size=8), automargin=True, row=2, col=1)

        
        return fig

        
    @render.ui  
    def model_explorer_content():  
        if data.get().empty:
            return ui.markdown("""
            ### 📊 Welcome to the Model Explorer 

            Please upload a CSV file to get started. The data should include these columns:
            - `stock_id`
            - `time_id`
            - `seconds_in_bucket`
            - `bid_price1`
            - `ask_price1`
            - `bid_price2`
            - `ask_price2`

            > NOTE: Make sure the CSV uses tab (`\\t`) as a delimiter.
                               
            ### Example:
 
            | stock_id | time_id | seconds_in_bucket | bid_price1 | ask_price1 | bid_price2 | ask_price2 |
            |----------|---------|-------------------|------------|------------|------------|------------|
            | 8382     | 12      | 1.0               | 722.17     | 722.63     | 722.15     | 722.64     |
            | 8382     | 12      | 2.0               | 722.18     | 722.88     | 722.17     | 722.98     |
            | ...      | ...     | ...               | ...        | ...        | ...        | ...        |
                               


            After loading, you'll be able to:
            - Select a stock and time ID
            - View model predictions and feature importance

            """)
        
        else:
            return output_widget("prediction")
    
    @render_widget  
    def prediction():  
        if get_vol_residual().empty or filtered_df().empty or input.timeid() is None:
            return None
        
        stock = get_vol_residual()[get_vol_residual()["time_id"] == int(input.timeid())]
        

        fig = go.Figure()
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['future'],
                    mode="lines",
                    line=dict(color=px.colors.qualitative.Plotly[0]), 
                    name='Realised Volatility',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['base_pred'],
                    mode="lines",
                    line=dict(color=px.colors.qualitative.Plotly[2]),
                    name='Predicted Volatility (Base)',
                )
            )
        fig.add_trace(
                go.Scatter(
                    x=stock["bucket"],
                    y=stock['base_pred'] + stock['vol_pred'],
                    mode="lines",
                    line=dict(color=px.colors.qualitative.Plotly[1]),
                    name='Predicted Volatility (Volume)',
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
    

    @render.ui
    def count():
        # Depend on data.  This is the key change.
        if data.get().empty:
            return ui.value_box(
                    "Number of timeIDs",
                    "0",
                    showcase=icon_svg("calendar"),
                ),
        df = data.get()
        return ui.value_box(
                    "Number of timeIDs",
                    str(df["time_id"].unique().shape[0]),
                    showcase=icon_svg("calendar"),
                ),

    @render.ui
    @reactive.event(base_runtime, vol_runtime)
    def traintime():
        if data.get().empty:
            return ui.value_box(
                ui.HTML(f"Model training time<br>(Final)"),
                "0.00 seconds",
                showcase=icon_svg("paper-plane"),
                theme="text-black",
            )

        return ui.value_box(
            ui.HTML(f"Model training time<br>(Final)"),
            f"{np.round(base_runtime.get() + vol_runtime.get(), 4)} seconds",
            showcase=icon_svg("paper-plane"),
            theme="text-green" if base_runtime.get() + vol_runtime.get() < 1 else "text-red",
        )

    @render.ui
    def directionalAccuracy():
        if get_vol_residual().empty:
            return ui.value_box(
                ui.HTML(f"Directional Accuracy Gain<br>(Base → Final)"),
                "0.00%",
                showcase=icon_svg("arrow-trend-up"),
                theme="text-black"
            )
        

        df = get_vol_residual()
        y_pred_base = df["base_pred"]
        y_pred_combined =df['base_pred'] + df['vol_pred']
        y_true = df['future']    

        # Compute direction of change
        base_diff = np.diff(y_pred_base)
        pred_diff = np.diff(y_pred_combined)
        real_diff = np.diff(y_true)

        # Directional accuracy: proportion of times model correctly predicts direction
        correct_direction = (pred_diff * real_diff) > 0
        directional_accuracy = np.mean(correct_direction) * 100

        correct_direction_base = (base_diff * real_diff) > 0
        directional_accuracy_base = np.mean(correct_direction_base) * 100

        directional_acc_increase = directional_accuracy - directional_accuracy_base


        icon = (
            "arrow-up" if directional_acc_increase > 0 else
            "arrow-down" if directional_acc_increase < 0 else
            "minus"
        )

        return ui.value_box(
            ui.HTML(f"Directional Accuracy Gain<br>(Base → Final)"),
            f"{directional_acc_increase:.2f}%",
            showcase=icon_svg(icon),
            theme="text-green" if directional_acc_increase > 0 else "text-red",
        )
    
    @render.ui
    @reactive.event(get_residual, get_vol_residual)
    def improvement():
        if get_vol_residual().empty:
            return ui.value_box(
                ui.HTML(f"RMSE Decrease<br>(Base → Final)"),
                "0.00%",
                showcase=icon_svg("bullseye"),
                theme="text-black"
            )
        
        vol_rmse = np.sqrt(np.mean(np.square(get_vol_residual()['vol_residual'])))
        base_rmse = np.sqrt(np.mean(np.square(get_residual()['residual'])))

        print(f"VOL RMSE: {vol_rmse}, BASE RMSE: {base_rmse}")

        decrease =  (base_rmse - vol_rmse) / base_rmse * 100
        print(f"Decrease: {decrease}")

        icon = (
            "arrow-up" if decrease > 0 else
            "arrow-down" if decrease < 0 else
            "minus"
        )

        return ui.value_box(
            ui.HTML(f"RMSE Decrease<br>(Base → Final)"),
            f"{decrease:.2f}%",
            showcase=icon_svg(icon),
            theme="text-green" if decrease > 0 else "text-red",
        )

    
    @render.ui
    def data_intro():

        md = ui.markdown(
            """
            # 📈 Volume-Volatilty Evaluation App

            ### Volume-Adjusted Residual Model with Volume Features
            Welcome to our Shiny app for evaluating the impact of volume-driven features to predict stock volatility using an *Volume-Adjusted Residual Model*. This tool is designed to assist traders in researching volume-based signals power on short-term price fluctuations.

            ---

            ## 🔍 What Is the Volume-Adjusted Residual Model?
            The *Volume-Adjusted Residual Model* enhances baseline volatility predictions by adjusting their residuals using XGBoost algorithms. These adjustments are informed by features derived from trading volume, allowing for more responsive and accurate volatility forecasts, particularly during periods of unusual market activity.

            ---

            ## 🧠 Features Used in the Model
            Our model uses the following engineered features, which capture various dimensions of market activity and price-volume interaction:
            - **ma5**: 5-period moving average of weighted average price.
            - **mid_price**:  Average of best bid and best ask prices
            - **volume_momentum_5**: Percentage change in 5-period volume moving average
            - **volume_trend**: Linear trend of volume over a 5-period window.
            - **volume_price_corr**: 5-period rolling correlation between volume and mid-price.
            - **vwap_deviation**: Deviation of current mid-price from the volume-weighted average price over 5 periods.
            - **order_flow_imbalance**: Normalized difference between bid and ask depth.
            - **cumulative_order_flow**: Rolling 5-period sum of order flow imbalance.
            - **volume_volatility**: Rolling standard deviation of volume.
            - **bs_volatility**: Volatility of bid-ask spread ratio over 5 periods.
            - **bs_momentum**: 3-period moving average of bid-ask spread changes.
            - **volume_ma_interaction**: Interaction term between volume moving average and a 5-period price moving average.
            - **bs_volume_interaction**:  Interaction term between bid-ask spread ratio and.

            ## ⚙️ App Functionality
            - *Visualisation*: Interactive plots for volatility predictions. 
            - *Model Diagnostics*: Residual analysis and performance metrics. 
            - *Performance Evaluation*: Training time, Directional Accuracy Increase and RMSEDecrease to evaluate the impact of volume model.
            - *Customization*: Option to choose different dataset, time id and features to predict and visualise.
            """,
        )

        return ui.div(md, class_="my-1")

    

app = App(app_ui, server)