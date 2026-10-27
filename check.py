import pandas as pd

orders = pd.read_csv("data/olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp",
                                  "order_delivered_customer_date"])
delivered = orders[orders["order_status"] == "delivered"]
days = (delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]).dt.days
print(round(days.mean(), 2))