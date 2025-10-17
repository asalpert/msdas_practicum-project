#region Load libraries
import numpy as np
import pandas as pd

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
#endregion

#region Read data
bep_df = pd.read_csv('data/BEP_data.csv.gz', compression='gzip', index_col=0)
#endregion

#region Data Preprocessing
def data_preprocess(df):
    # Rename columns
    df.rename(columns={'Unnamed: 0': 'timestamp',
                       '1. open': 'open',
                       '2. high': 'high',
                       '3. low': 'low',
                       '4. close': 'close',
                       '5. volume': 'volume'}, inplace=True)
    
    # Drop unnecessary columns
    if 'Unnamed: 0.1' in df.columns.tolist():
        df.drop(columns=['Unnamed: 0.1'], inplace=True)

    # Handle data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    return df

bep_df = data_preprocess(bep_df)

#endregion

#region Data Summary

# Get some summary statistics on the data: mean, median, min, max, standard deviation
bep_summary = bep_df.describe()

# Plots
def plot_timeseries(df, column):
    g = (ggplot(df) + 
         geom_line(aes(x='timestamp', y=column)) + 
         labs(title=column))
    return g

plot_timeseries(bep_df, 'open')
plot_timeseries(bep_df, 'close')
plot_timeseries(bep_df, 'high')
plot_timeseries(bep_df, 'low')
#endregion

#region Volatility checks
def resample_data(df, frequency):
    agg_funcs = {'open': 'mean',
                 'close': 'mean',
                 'high': 'max',
                 'low': 'min',
                 'volume': 'mean'}
    
    df = df.set_index('timestamp')

    resampled_df = df.resample(frequency).agg(agg_funcs)
    
    resampled_df = resampled_df.reset_index()

    return resampled_df

def volatility_check(resampled_df):
    resampled_df['open-close'] = resampled_df['open'] - resampled_df['close']
    resampled_df['high-low'] = resampled_df['high'] - resampled_df['low']

    volatiltiy_summary = resampled_df.describe()

    oc_plot = (ggplot(resampled_df) + 
         geom_line(aes(x='timestamp', y='open-close')) + 
         labs(title='Difference between open and close prices'))
    
    hl_plot = (ggplot(resampled_df) + 
         geom_line(aes(x='timestamp', y='high-low')) + 
         labs(title='Difference between high and low prices'))
    
    return volatiltiy_summary, oc_plot, hl_plot



#endregion