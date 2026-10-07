#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd 
from pathlib import Path 
import geopandas as geopd
import plotly.graph_objs as go 
import numpy as np 
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

def plot_overlapped(df, col_data='date', col_comune='comune', col_valore='value'):
    fig = go.Figure()

    # Raggruppiamo per comune per aggiungere una traccia (linea) ciascuno
    for comune, group in df.groupby(col_comune):
        fig.add_trace(
            go.Scatter(
                x=group[col_data],
                y=group[col_valore],
                mode='lines',
                name=comune,
                text=group[col_comune],
                hovertemplate='<b>%{text}</b><br>Data: %{x|%Y-%m-%d}<br>Presenze: %{y:,}<extra></extra>'
            )
        )

    fig.update_layout(
        title="Presenze Turistiche per Comune",
        xaxis_title="Data",
        yaxis_title="Presenze (Turisti)",
        hovermode="closest",
        template="plotly_white",
        legend_title="Comuni"
    )

    return fig

def plot_stats(df, col_data='date', col_comune='comune', col_valore='value'):
    """Overlaps and stats"""
    df = df.copy()
    df[col_data] = pd.to_datetime(df[col_data])
    
    stats_df = df.groupby(col_data)[col_valore].agg(['mean', 'median', 'max']).reset_index().sort_values(col_data)
    
    fig = go.Figure()
    
    comuni = df[col_comune].unique()
    for i, comune in enumerate(comuni):
        df_comune = df[df[col_comune] == comune].sort_values(col_data)

        # Mostriamo la voce in legenda SOLO al primo giro del ciclo
        mostra_nella_legenda = True if i == 0 else False

        fig.add_trace(go.Scatter(
            x=df_comune[col_data],
            y=df_comune[col_valore],
            mode='lines',
            name='Singoli Comuni',              # Nome per la legenda
            legendgroup='group_comuni',         # Raggruppa tutte queste linee
            showlegend=mostra_nella_legenda,
            line=dict(color='rgba(31, 119, 180, 0.15)', width=1), # Sottili e semitrasparenti (azzurro)
            hovertemplate=f"<b>{comune}</b><br>Presenze: %{{y}}<extra></extra>" 
        ))
    # MEDIA 
    fig.add_trace(go.Scatter(
        x=stats_df[col_data],
        y=stats_df['mean'],
        mode='lines',
        name='Media (per Comune)',
        line=dict(color='rgba(214, 39, 40, 1.0)', width=3), # Rosso fuoco, spessa
        hovertemplate="<b>Media</b><br>Presenze: %{y:.0f}<extra></extra>"
    ))

    # 4. MEDIANA
    fig.add_trace(go.Scatter(
        x=stats_df[col_data],
        y=stats_df['median'],
        mode='lines',
        name='Mediana (per Comune)',
        line=dict(color='yellow', width=2), 
        hovertemplate="<b>Mediana</b><br>Presenze: %{y:.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Andamento Presenze Turistiche: Distribuzione Comuni vs Statistiche",
        xaxis_title="Data",
        yaxis_title="Presenze Turistiche",
        hovermode='closest', # Usiamo 'closest' invece di 'x unified' per non intasare lo schermo con 160+ tooltip
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()


# %%
## PLOT: tourist attendences 
fig = plot_overlapped(tourists_attendences)
fig.show()
# %%
## PLOT: tourist attendences 
plot_stats(tourists_attendences)

# %% [markdown]
# # Variational analysis 
# Here we calculate the "derivative" to see the variation 
# 
# **Note:** Dropping `NaN` to avoid errors.

tourists_attendences['value_diff'] = tourists_attendences['value'].diff()
plot_stats(tourists_attendences.dropna(), col_valore = "value_diff")

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
