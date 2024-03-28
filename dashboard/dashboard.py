import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

def create_aq_mean_df(df):
    aq_mean_df = pd.DataFrame({
        'station_name': ['changping_station', 'dingling_station', 'dongsi_station', 'shunyi_station'],
        'PM2.5': [df['PM2.5_x_x'].mean(), df['PM2.5_y_x'].mean(), df['PM2.5_x_y'].mean(), df['PM2.5_y_y'].mean()],
        'PM10': [df['PM10_x_x'].mean(), df['PM10_y_x'].mean(), df['PM10_x_y'].mean(), df['PM10_y_y'].mean()]
    })

    return aq_mean_df

def create_pm25_trend_df(df):
    pm25_trend_df = pd.DataFrame({
        'Changping Station': df['PM2.5_x_x'],
        'Dingling Station': df['PM2.5_y_x'],
        'Dongsi Station': df['PM2.5_x_y'],
        'Shunyi Station': df['PM2.5_y_y']
        })
    pm25_trend_df.columns.name='Station'
    pm25_trend_df.index.name='year'
    return pm25_trend_df

def create_pm10_trend_df(df):
    pm10_trend_df = pd.DataFrame({
        'Changping Station': df['PM10_x_x'],
        'Dingling Station': df['PM10_y_x'],
        'Dongsi Station': df['PM10_x_y'],
        'Shunyi Station': df['PM10_y_y']
        })
    pm10_trend_df.columns.name='Station'
    pm10_trend_df.index.name='year'
    return pm10_trend_df

def create_so2_trend_df(df):
    so2_trend_df = pd.DataFrame({
        'Changping Station': df['SO2_x_x'],
        'Dingling Station': df['SO2_y_x'],
        'Dongsi Station': df['SO2_x_y'],
        'Shunyi Station': df['SO2_y_y']
        })
    so2_trend_df.columns.name='Station'
    so2_trend_df.index.name='year'
    return so2_trend_df

def create_no2_trend_df(df):
    no2_trend_df = pd.DataFrame({
        'Changping Station': df['NO2_x_x'],
        'Dingling Station': df['NO2_y_x'],
        'Dongsi Station': df['NO2_x_y'],
        'Shunyi Station': df['NO2_y_y']
        })
    no2_trend_df.columns.name='Station'
    no2_trend_df.index.name='year'
    return no2_trend_df

def create_co_trend_df(df):
    co_trend_df = pd.DataFrame({
        'Changping Station': df['CO_x_x'],
        'Dingling Station': df['CO_y_x'],
        'Dongsi Station': df['CO_x_y'],
        'Shunyi Station': df['CO_y_y']
        })
    co_trend_df.columns.name='Station'
    co_trend_df.index.name='year'
    return co_trend_df

def create_o3_trend_df(df):
    o3_trend_df = pd.DataFrame({
        'Changping Station': df['O3_x_x'],
        'Dingling Station': df['O3_y_x'],
        'Dongsi Station': df['O3_x_y'],
        'Shunyi Station': df['O3_y_y']
        })
    o3_trend_df.columns.name='Station'
    o3_trend_df.index.name='year'
    return o3_trend_df

main_df = pd.read_csv('main_data.csv')

aq_mean_df = create_aq_mean_df(main_df)
pm25_trend_df = create_pm25_trend_df(main_df)
pm10_trend_df = create_pm10_trend_df(main_df)
so2_trend_df = create_so2_trend_df(main_df)
no2_trend_df = create_no2_trend_df(main_df)
co_trend_df = create_co_trend_df(main_df)
o3_trend_df = create_o3_trend_df(main_df)

st.header("Air Quality Analysis")

st.subheader('Station with the Highest Air Pollution')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

colors = ['#B0C4DE', '#B0C4DE', '#00008B', '#B0C4DE']

sns.barplot(x='station_name', y='PM2.5', data=aq_mean_df, palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title(r"Based on PM$_2$$_.$$_5$ Concentration", loc='center', fontsize=30)
ax[0].tick_params(axis ='x', labelsize=12)

sns.barplot(x='station_name', y='PM10', data=aq_mean_df, palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title(r"Based on PM$_1$$_0$ Concentration", loc='center', fontsize=30)
ax[1].tick_params(axis ='x', labelsize=12)

st.pyplot(fig)

st.subheader("Air Pollution Trends")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20,10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(pm25_trend_df.columns):
        color = cmap(i / len(pm25_trend_df.columns))
        plt.plot(pm25_trend_df.index, pm25_trend_df[column], color=color, label=column)
    ax.set_title(r"Based on PM$_2$$_.$$_5$ Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(20,10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(pm10_trend_df.columns):
        color = cmap(i / len(pm10_trend_df.columns))
        plt.plot(pm10_trend_df.index, pm10_trend_df[column], color=color, label=column)
    ax.set_title(r"Based on PM$_1$$_0$ Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(20,10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(so2_trend_df.columns):
        color = cmap(i / len(so2_trend_df.columns))
        plt.plot(so2_trend_df.index, so2_trend_df[column], color=color, label=column)
    ax.set_title(r"Based on SO$_2$ Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots(figsize=(20,10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(no2_trend_df.columns):
        color = cmap(i / len(no2_trend_df.columns))
        plt.plot(no2_trend_df.index, no2_trend_df[column], color=color, label=column)
    ax.set_title(r"Based on NO$_2$ Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(co_trend_df.columns):
        color = cmap(i / len(co_trend_df.columns))
        plt.plot(co_trend_df.index, co_trend_df[column], color=color, label=column)
    ax.set_title("Based on CO Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

with col6:
    fig, ax = plt.subplots(figsize=(20,10))
    cmap = plt.cm.get_cmap('viridis')
    for i, column in enumerate(o3_trend_df.columns):
        color = cmap(i / len(o3_trend_df.columns))
        plt.plot(o3_trend_df.index, o3_trend_df[column], color=color, label=column)
    ax.set_title(r"Based on O$_3$ Concentration", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    ax.legend()
    st.pyplot(fig)

st.caption('Copyright (c) Dicoding 2023')