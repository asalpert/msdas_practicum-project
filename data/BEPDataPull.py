#region Libraries
import os
import requests
import json
import gzip
import pandas as pd
import glob
#endregion

#region Gobals
companies = ['LLY', 'APPL', 'BEP']
params = {'function': 'TIME_SERIES_INTRADAY',
          'symbol': 'BEP',
          'interval': '1min',
          'month': '2023-01',
          'outputsize': 'full',
          'extended_hours': 'false',
          'apikey': os.environ["API_KEY"]
          }
#endregion

#region Prep 2024 API call
yyyy = '2020'
mm = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
#endregion

#region 2024 API call
for m in mm:
    params['month']=yyyy+"-"+m
    url = "https://www.alphavantage.co/query?"
    for key, val in params.items():
        url+=key
        url+='='
        url+=val
        url+='&'
    url = url[:-1]
    r = requests.get(url)
    data = r.json()
    print(m, len(data['Time Series (1min)']))
    df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
    df = df.sort_index()
    # print(df)
    df.to_csv(params['symbol'] + params['month'] + '.csv.gz', compression='gzip')
#endregion
  

#region Prep 2025 API call
yyyy = '2021'
mm = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
#endregion

#region 2025 API call
for m in mm:
    params['month']=yyyy+"-"+m
    url = "https://www.alphavantage.co/query?"
    for key, val in params.items():
        url+=key
        url+='='
        url+=val
        url+='&'
    url = url[:-1]
    r = requests.get(url)
    data = r.json()
    print(m, len(data['Time Series (1min)']))
    df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
    df = df.sort_index()
    # print(df)
    df.to_csv(params['symbol'] + params['month'] + '.csv.gz', compression='gzip')
#endregion
    
#region Concat into one frame per stock
# Read in all CSVs for the stock
csv_files = glob.glob(os.path.join(os.getcwd(), f"*{params['symbol']}*.csv.gz"))

combined_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
#endregion