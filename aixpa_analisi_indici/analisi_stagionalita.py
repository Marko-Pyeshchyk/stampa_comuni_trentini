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

from help_plots import plot_overlapped, plot_stats

import sys
sys.path.insert(0, str(Path("/home/aricci/stampa_comuni_trentini").resolve()))
from utils import get_s3, get_mapping, get_dataframe

# %% [markdown]
## Download of the data and processing 
#### Note: only the TOURISTS are taken into account in the following analysis 

# In[ ]:

vodafone_attendences = get_dataframe("vodafone_attendences")
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

# Scomposizione MSTL 
# Impostiamo la stagionalità settimanale (7 giorni) e annuale (365 giorni)
# Se manca al massimo una settimana di dati, ricostruisco i valori sulla base dei giorni circostanti.

s_mstl = (
    t.pivot(index="date", columns="comune", values="value")
    .reindex(pd.date_range("2022-01-01", "2025-12-31", freq="D"))
)
s_mstl = s_mstl.interpolate(method="time", limit=7)
s_mstl.isna().sum().sort_values(ascending=False).head(20)

ts_comune = s_mstl[comune_test].dropna()

mstl = MSTL(
    ts_comune,
    periods=(7, 365),
    iterate=3
)

res = mstl.fit()

# %% 
mstl = MSTL(ts_comune, periods=(7, 30), iterate=3)
res = mstl.fit()

ts_trend_season = res.trend + res.seasonal['seasonal_30']

fig_smooth.add_trace(go.Scatter(
    x=ts_trend_season.index, y=ts_trend_season,
    mode='lines', name='Trend + Stagionalità Annuale (MSTL)',
    line=dict(color='red', width=2)
))

resid = res.resid.dropna()

fig, ax = plt.subplots(figsize=(15, 4))

ax.plot(resid)

ax.set_title(f"Residui MSTL - {comune_test}")
ax.set_xlabel("Data")
ax.set_ylabel("Residuo")

plt.tight_layout()
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(resid, lags=100, ax=ax)

ax.set_title(f"ACF dei residui MSTL - {comune_test}")

plt.tight_layout()
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(resid, lags=100, ax=ax)

ax.set_title(f"ACF dei residui MSTL - {comune_test}")

plt.tight_layout()
plt.show()

# %% [markdown]
# APPENDIX

# CONTROLLI EXTRA 
s = tourists_attendences["value"].unstack("comune")
s.index = pd.to_datetime(s.index)
ids = tourists_attendences["ID_COMUNE"].groupby(level="comune").first()

# Integrità
full_range = pd.date_range("2022-01-01", "2025-12-31", freq="D")
print("comuni:", s.shape[1], "| giorni:", s.shape[0], "| attesi:", len(full_range))
print("duplicati indice:", tourists_attendences.index.duplicated().sum())
print("giorni mancanti:", full_range.difference(s.index).size)

# 3) NaN, zeri, negativi per comune
qc = pd.DataFrame({
    "n_nan":   s.isna().sum(),
    "pct_zero": (s.eq(0)).mean().round(3),
    "n_neg":   (s < 0).sum(),
    "tot":     s.sum(),
    "max":     s.max(),
    "max_over_median": (s.max() / s.median().replace(0, np.nan)).round(1),
})
print(qc.sort_values("pct_zero", ascending=False).head(20))
print(qc.query("n_nan > 0 or n_neg > 0"))

# 4) totale per anno e comune (cerca anni vuoti o salti strani)
yearly = s.groupby(s.index.year).sum().T
yearly["ratio_25_22"] = yearly[2025] / yearly[2022]
print(yearly.sort_values("ratio_25_22").head(10))
print(yearly.sort_values("ratio_25_22").tail(10))
# %%
