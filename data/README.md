# Data Instructions

The project uses the IBM Telco Customer Churn dataset.

## Automatic download

Run:

```bash
python data/download_data.py
```

This will try to download the CSV from:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Manual download fallback

If the raw URL is unavailable, download the dataset from Kaggle and save it as:

`data/Telco-Customer-Churn.csv`

Kaggle link:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn
