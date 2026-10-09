#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd 
from pathlib import Path 
import geopandas as geopd
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt 
import numpy as np 

from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path("/home/aricci/stampa_comuni_trentini").resolve()))
from utils import get_s3, get_mapping, get_dataframe
from aixpa_analisi_indici.help_plots import plot_overlapped, plot_stats

# %% [markdown]
## Download of the data and processing 
#### Note: only the TOURISTS are taken into account in the following analysis 

# In[ ]:

vodafone_attendences = pd.read_csv(get_s3("vodafone_attendences.csv"))
vodafone_attendences_new = pd.read_csv(get_s3("vodafone_attendences_new.csv"))

geojson_comuni_json_data = geopd.read_file(get_s3("TRENTINO-comuni_Vodafone_2023.geojson"))
json_vodafone = get_mapping("mapping_comuni_into_vodafone_Trento.json")
json_apt = get_mapping("map_comuni_into_apt.json")


# In[ ]:
location_map = dict(
    zip(
        geojson_comuni_json_data["id"].astype(str),
        geojson_comuni_json_data["name"].str.upper(),
    )
)
# geojson_comuni_json_data[geojson_comuni_json_data.name != geojson_comuni_json_data.desc]

vodafone_attendences['date'] = pd.to_datetime(vodafone_attendences['date'])
vodafone_attendences_new['date'] = pd.to_datetime(vodafone_attendences_new['date'])
vodafone_attendences = vodafone_attendences[~vodafone_attendences['date'].isin(vodafone_attendences_new['date'])]

# merge 
vodafone_attendences_merged = pd.concat([vodafone_attendences, vodafone_attendences_new], ignore_index=True)
vodafone_attendences_merged = vodafone_attendences_merged.sort_values('date').reset_index(drop=True)

vodafone_attendences_merged_comuni = vodafone_attendences_merged[
    vodafone_attendences_merged["locType"] == "TN_MKT_AL_3"
]

vodafone_attendences_merged_comuni["comune"] = vodafone_attendences_merged_comuni["locId"].astype(str).map(location_map)
vodafone_attendences_merged_comuni["ID_COMUNE"] = vodafone_attendences_merged_comuni["comune"].map(json_vodafone)
mask = vodafone_attendences_merged_comuni["comune"].isin(["VIGO DI FASSA", "POZZA DI FASSA"])
vodafone_attendences_merged_comuni.loc[mask, "comune"] = "SAN GIOVANNI DI FASSA" # Vigo di Fassa e Pozza di Fassa -> San Giovanni di Fassa
vodafone_attendences_merged_comuni.loc[mask, "ID_COMUNE"] = pd.Series(
    [[22250]] * mask.sum(), index=vodafone_attendences_merged_comuni.index[mask]  
)

tourists_attendences = vodafone_attendences_merged_comuni[vodafone_attendences_merged_comuni['userProfile'] == 'TOURIST'].groupby(['date', 'comune']).agg({'value': 'sum', 'ID_COMUNE':'first'}).reset_index()
tourists_attendences


# %%
## checks 
vodafone_attendences_merged_comuni.ID_COMUNE.isna().any()
vodafone_attendences_merged_comuni.comune.isna().any()

# %% [markdown]
#### Here, we define the helper functions to use in plotting 


# %%
## PLOT: tourist attendences 
fig = plot_overlapped(tourists_attendences)
fig.show()
## PLOT: tourist attendences
fig = plot_stats(tourists_attendences)
fig.show()

# %% 
## Add rolling window
t = tourists_attendences.copy()
s_clean = (
    t.pivot(index="date", columns="comune", values="value")
    .reindex(pd.date_range("2022-01-01", "2025-12-31", freq="D")) # 32 comuni tt.isna().sum()[tt.isna().sum() > 0].sort_values()
    .fillna(0)
)
# Comune di esempio
comune_test = "PINZOLO"
ts_comune = s_clean[comune_test]

ts_rolling_7 = ts_comune.rolling(window=7, center=True).mean()
ts_rolling_14 = ts_comune.rolling(window=14, center=True).mean()
ts_rolling_20 = ts_comune.rolling(window=20, center=True).mean()

fig_smooth = go.Figure()

fig_smooth.add_trace(go.Scatter(
    x=ts_comune.index, y=ts_comune,
    mode='lines', name='Dati Grezzi (Giornalieri)',
    line=dict(color="#676867", width=1)
))

fig_smooth.add_trace(go.Scatter(
    x=ts_rolling_7.index, 
    y=ts_rolling_7,
    mode='lines',
    name='Rolling 7 gg',
    line=dict(color="#278324", width=2) 
))

fig_smooth.add_trace(go.Scatter(
    x=ts_rolling_14.index, 
    y=ts_rolling_14,
    mode='lines',
    name='Rolling 14 gg',
    line=dict(color="#51AFC5", width=2) 
))

fig_smooth.add_trace(go.Scatter(
    x=ts_rolling_20.index, 
    y=ts_rolling_20,
    mode='lines',
    name='Rolling 20 gg',
    line=dict(color="#B549E7", width=2) 
))

fig_smooth.update_layout(
    title=f"Smoothing - Comune di {comune_test}",
    xaxis_title="Data",
    yaxis_title="Presenze Turistiche",
    template="plotly_white",
    hovermode="x unified"
)
fig_smooth.show()

# %% 
# PEAK ANALYSIS

y_smoothed = ts_rolling_14.dropna() # Rimuoviamo i NaN generati dal rolling

# find_peaks trova gli indici (la posizione) dei picchi
# 'distance=14' impone che ci siano almeno 14 giorni tra un picco e l'altro
# 'prominence=500' impone che il picco si innalzi di almeno 500 presenze rispetto alle valli circostanti
peaks_indices, _ = find_peaks(y_smoothed, distance=14, prominence=500)

# Otteniamo le date e i valori reali dei picchi
date_picchi = y_smoothed.index[peaks_indices]
valori_picchi = y_smoothed.iloc[peaks_indices]

# Aggiungiamo i picchi al tuo grafico Plotly esistente (fig_smooth)
fig_smooth.add_trace(go.Scatter(
    x=date_picchi, 
    y=valori_picchi,
    mode='markers',
    name='Picchi Principali',
    marker=dict(color='red', size=5, symbol='circle', line=dict(color='black', width=.5)),
    hovertemplate="<b>Picco</b><br>Data: %{x|%Y-%m-%d}<br>Presenze: %{y:.0f}<extra></extra>"
))

fig_smooth.show()

