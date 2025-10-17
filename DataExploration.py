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
                 'volume': 'sum'}
    
    df = df.set_index('timestamp')

    resampled_df = df.resample(frequency).agg(agg_funcs)
    
    resampled_df = resampled_df.reset_index()

    return resampled_df

def volatility_check(df, frequency):
    
    resampled_df = resample_data(df, frequency)

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


time_intervals = ['1H', '1D', 'W', '1M', 'Q', 'Y'] # day, month, week, month, quarter, year

bep_oc_stats = []
bep_hl_stats = []
bep_volatity_plots = []

for inter in time_intervals:
    bep_summary, bep_oc, bep_hl = volatility_check(bep_df, inter)

    bep_oc_mean = bep_summary['open-close']['mean']
    bep_oc_max = bep_summary['open-close']['max']

    bep_hl_mean = bep_summary['high-low']['mean']
    bep_hl_max = bep_summary['high-low']['max']

    bep_oc_stats.append({'interval': inter, 'mean': bep_oc_mean, 'max': bep_oc_max})
    bep_hl_stats.append({'interval': inter, 'mean': bep_hl_mean, 'max': bep_hl_max})
    bep_volatity_plots.append({'interval': inter, 'open-close': bep_oc, 'high-low': bep_hl})




#endregion