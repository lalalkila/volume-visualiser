import seaborn as sns
from faicons import icon_svg
import pandas as pd
import numpy as np

# Import data from shared.py
from shared import app_dir, df

from shiny import App, reactive, render, ui

data = reactive.Value(df)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload a CSV file", accept=".csv"),
        ui.input_select(
            "timeid",
            "Time ID",
            [],
            selected=None,
        ),
    ),
    ui.layout_column_wrap(
        ui.value_box(
            "Number of timeIDs",
            ui.output_text("count"),
            showcase=icon_svg("earlybirds"),
        ),
        ui.value_box(
            "Number of timeIDs",
            ui.output_text("count1"),
            showcase=icon_svg("earlybirds"),
        ),
        ui.value_box(
            "Number of timeIDs",
            ui.output_text("count3"),
            showcase=icon_svg("earlybirds"),
        ),
        fill=False,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Features explorer"),
            ui.output_plot("length_depth"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Data explorer"),
            ui.output_data_frame("summary_statistics"),
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
            df = pd.read_csv(file_info["datapath"])
            df['bucket'] = np.floor(df['seconds_in_bucket'] / 30)
            df = df.groupby(['time_id', 'bucket']).mean()[['bid_price1','ask_price1','bid_price2','ask_price2','bid_size1','ask_size1','bid_size2', 'ask_size2']].round(4).reset_index()
            data.set(df)
            ui.update_select(
                "timeid",
                label="Choose TimeID:",
                choices=df["time_id"].unique().tolist(),
                selected=df["time_id"].unique()[0],
            )

    @render.data_frame
    def summary_statistics():
        return filtered_df()

    @reactive.calc
    def filtered_df():
        df = data.get()
        # if input.timeid() and not df.empty:
        #     return df[df["time_id"] == input.timeid()]
        return df

    @render.text
    def count():
        return data.get()["time_id"].unique().shape[0]

    @render.text
    def count2():
        return str(input.timeid())

app = App(app_ui, server)
