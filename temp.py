import pandas as pd

df = pd.read_csv("W:/UNI/DATA3888/Optiver_additional data/Optiver_additional data/order_book_feature.csv", delimiter="\t")

df[df['stock_id'] == 8382].to_csv("8382.csv", index=False, sep="\t")