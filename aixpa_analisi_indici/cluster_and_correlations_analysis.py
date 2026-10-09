#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
from pathlib import Path 
import geopandas as geopd
import plotly.graph_objs as go 
import plotly.figure_factory as ff
import plotly.express as px 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path("/home/aricci/stampa_comuni_trentini").resolve()))
from utils import get_s3, get_mapping
from aixpa_analisi_indici.help_plots import plot_overlapped, plot_stats, find_peaks_dynamic

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


# %%
## ANALISI DEI VALORI MANCANTI  

t = tourists_attendences.copy()
s = t.pivot(index="date", columns="comune", values="value")
corr = s.corr()

print("NaN totali:", s.isna().sum().sum())
print("\nNaN per comune:")
print(s.isna().sum().sort_values(ascending=False))

missing = (
    s.isna()
     .sum()
     .sort_values(ascending=False)
)
missing_pct = (missing / len(s) * 100).round(2)
df_nans = pd.DataFrame({
    "n_nan": missing,
    "perc_nan": missing_pct
}).head(15)

df_nans
# %% 
# %% 
nan_counts = s.isna().sum()
comuni_con_nan = sorted(nan_counts[nan_counts > 0].index)

if len(comuni_con_nan) == 0:
    print("Nessun comune presenta valori mancanti.")
else:
    s_nan = s[comuni_con_nan].isna()
    s_nan_matrix = s_nan.T

    plt.figure(figsize=(18, max(6, len(comuni_con_nan) * 0.4)))
    plt.imshow(
        s_nan_matrix,
        aspect="auto",
        interpolation="none"
    )
    plt.yticks(
        range(len(s_nan_matrix.index)),
        s_nan_matrix.index
    )
    dates = s_nan_matrix.columns
    n_ticks = 12
    idx = np.linspace(0, len(dates) - 1, min(n_ticks, len(dates)), dtype=int)

    plt.xticks(
        idx,
        [dates[i].strftime("%Y-%m-%d") for i in idx],
        rotation=45
    )
    plt.xlabel("Data")
    plt.ylabel("Comune")
    plt.title(f"Distribuzione dei valori mancanti nel tempo ({len(comuni_con_nan)} comuni con NaN)")

    plt.colorbar(label="NaN (1 = mancante)")
    plt.tight_layout()
    plt.show()


# %% 
# Ci sono due strade, tenere i nan o fillarli con 0. In ogni caso spoiler il comportamnto sembra cambiare poco 
## la serie presenta diversi buchi, per i comuni piu' piccoli ci sono diverse date mancanti. 
# Filliamo, per il momento con degli zeri 

s_clean = (
    s
    .reindex(pd.date_range("2022-01-01", "2025-12-31", freq="D")) # 32 comuni tt.isna().sum()[tt.isna().sum() > 0].sort_values()
    .fillna(0)
)
corr_matrix = s_clean.corr()

# %%
## DENDOGRAMMI 

## Confronto tra 
# - i 3 metodi
# - le due matrici di correlaizoni

dist_matrix = 1 - corr_matrix
condensed_dist = squareform(dist_matrix.values, checks=False)

for method in ["average", "complete", "single", "ward"]:
    Z_test = linkage(condensed_dist, method=method)
    
    for k in range(2, 8):
        labels = fcluster(Z_test, t=k, criterion="maxclust")
        # Per silhouette_score possiamo passare la matrice quadrata dist_matrix
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        print(f"{method:8s} | k={k} | silhouette={score:.3f}")

# %%
CLUSTERS = 3
METHOD = "ward"
CORRMX = corr 

# Clustering con metodo Ward
dist_matrix = 1 - corr_matrix
condensed_dist = squareform(dist_matrix.values, checks=False)
Z = linkage(dist_matrix, method=METHOD)

dendogram = ff.create_dendrogram(
    dist_matrix, 
    labels=CORRMX.columns.tolist(),
    linkagefun=lambda x: Z
)

dendogram.update_layout(
    title=f"Dendrogramma della Similarità Turistica tra Comuni, metodo {METHOD}",
    xaxis_title="Comuni",
    yaxis_title="Distanza",
    width=2000,
    height=700,
    template="plotly_white"
)

dendogram.update_xaxes(tickangle=-45)
dendogram.show()

# 5. Estrazione dei 4 Cluster
clusters = fcluster(Z, t=CLUSTERS, criterion='maxclust')

df_clusters = pd.DataFrame({
    'comune': CORRMX.columns,
    'cluster': clusters
}).sort_values('cluster')

print(f"--- Distribuzione Comuni nei {CLUSTERS} Cluster ---")
print(df_clusters['cluster'].value_counts())

# %%
rolling = 14 
for cc in df_clusters['cluster'].unique():
    clust_subdf = df_clusters[df_clusters['cluster'] == cc]
    ## SUBDF OF ATTENDENCES OF COMUNI IN CLUSTER 
    tourists_attendences_clust_subdf = tourists_attendences[tourists_attendences['comune'].isin(clust_subdf['comune'].unique())]

    ## ROLLING WINDOW 
    s_clean = (
        tourists_attendences_clust_subdf.pivot(index="date", columns="comune", values="value")
        .reindex(pd.date_range("2022-01-01", "2025-12-31", freq="D")) # 32 comuni tt.isna().sum()[tt.isna().sum() > 0].sort_values()
        .fillna(0)
    )
    ts_rolling_14 = s_clean.rolling(window=rolling, center=True).mean()  
    ts_normalized = (ts_rolling_14 - ts_rolling_14.min()) / (ts_rolling_14.max() - ts_rolling_14.min())

    ## NORMALIZATION
    ts_normalized_long = (
        ts_normalized.reset_index()
        .melt(id_vars="index", var_name="comune", value_name="value")
        .rename(columns={"index": "date"}) 
        .dropna() 
    )

    ## FIND PEAKS 
    fig = plot_stats(ts_normalized_long)

    all_peak_dates = []
    for comune in ts_normalized.columns:
        ts_comune = ts_normalized[comune].dropna()
        peaks_idx, _ = find_peaks(ts_comune, distance=14, prominence=0.15)
        all_peak_dates.extend(ts_comune.index[peaks_idx])

        
        date_picchi = ts_comune.index[peaks_idx]
        valori_picchi = ts_comune.iloc[peaks_idx]

        fig.add_trace(go.Scatter(
            x=date_picchi,
            y=valori_picchi,
            mode='markers',
            name=f'Picchi {comune}',
            marker=dict(size=5, symbol='circle', color = "blue"),
            showlegend=False,  
            hovertemplate=f"<b>Picco ({comune})</b><br>Data: %{{x|%Y-%m-%d}}<br>Valore Norm: %{{y:.2f}}<extra></extra>"
        ))

    df_peaks = pd.DataFrame({'date': all_peak_dates})

    fig.update_layout(
        title=f"Cluster {cc} - Trend Normalizzato e Picchi Principali (Rolling {rolling} gg)",
        yaxis_title="Presenze Normalizzate (0=Min, 1=Max)"
    )
    for d in date_picchi:
        fig.add_vline(
            x=d, 
            line_dash="dot", 
            line_color="blue", 
            line_width=1,
            opacity=0.6
        )
    fig.show()
    
    fig_hist = px.histogram(
        df_peaks, 
        x='date', 
        nbins=120,  
        title=f"Cluster {cc} - Concentrazione delle Date di Picco tra i Comuni",
        labels={'date': 'Data', 'count': 'N° Comuni in Picco'},
        template='plotly_white'
    )

    fig_hist.update_traces(marker_color='#1f77b4')
    fig_hist.update_layout(bargap=0.1)
    fig_hist.show()

# %%
