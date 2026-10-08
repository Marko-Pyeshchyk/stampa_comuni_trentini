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
import pandas as pd 

from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

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


for method in ["average", "complete", "single", "ward"]:
    dist_matrix = 1 - corr
    condensed_dist = squareform(dist_matrix, checks=False)
    Z_test = linkage(dist_matrix, method=method)
    
    for k in range(2, 8):
        labels = fcluster(Z_test,t=k,criterion="maxclust")
        score = silhouette_score(dist_matrix,labels,metric="precomputed")
        print(f"{method:8s} | k={k} | silhouette={score:.3f}")

# %%
CLUSTERS = 4
METHOD = "ward"
CORRMX = corr 

# Clustering con metodo Ward
dist_matrix = 1 - corr
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
