import geopandas as gpd 
import numpy as np 
import matplotlib.pyplot as plt 
from shapely.ops import linemerge
from shapely import union_all
import sys 
from shapely.geometry import Point
import itertools 

## ===== SPECIFIC FUNCTIONS FOR TRAFFICO VEICOLARE ======

def compute_loc_spira(row, gdf_strade, km_inizio_dict):
    strada = row['STRADA']  
    km_spira = row['KM']        
    
    if strada in gdf_strade.index:
        geom_strada = gdf_strade.loc[strada, 'geometry']
        km_inizio_strada = km_inizio_dict.get(strada, 0.0)
        
        dist_spira = km_spira - km_inizio_strada
        distanza_metri = dist_spira * 1000 # Convertiamo in metri 
        
        if distanza_metri >= 0 and distanza_metri <= geom_strada.length:
            return geom_strada.interpolate(distanza_metri)
    return None

def create_map_start_end(gdf_spatial, df_tabular):
    gdf_spatial = gdf_spatial.copy()
    df_tabular = df_tabular.copy()

    gdf_spatial["str_cd"] = gdf_spatial["str_cd"].astype(str).str.strip()
    df_tabular["id_lrs"] = df_tabular["id_lrs"].astype(str).str.strip()

    gdf_merged = gdf_spatial.merge(
        df_tabular, left_on="str_cd", right_on="id_lrs", how="left"
    )

    starts, ends = [], []
    start_labels, end_labels = [], []

    # Raggruppiamo per strada e calcoliamo i punti 
    for str_cd, group in gdf_merged.groupby("str_cd"):
        start, end = compute_start_end(group)
        starts.append(Point(start))
        ends.append(Point(end))

        # Recuperiamo i dati testuali 
        denominazione = (
            group["denominazione"].iloc[0]
            if "denominazione" in group.columns
            else ""
        )
        cod_servizio = (
            group["codice servizio"].iloc[0]
            if "codice servizio" in group.columns
            else str_cd
        )

        loc_inizio = (
            group["localita di inizio strada"].iloc[0]
            if "localita di inizio strada" in group.columns
            else "Inizio sconosciuto"
        )
        km_inizio = (
            group["progressiva di inizio strada"].iloc[0]
            if "progressiva di inizio strada" in group.columns
            else "0"
        )

        loc_fine = (
            group["localita di fine strada"].iloc[0]
            if "localita di fine strada" in group.columns
            else "Fine sconosciuta"
        )
        km_fine = (
            group["progressiva di fine strada"].iloc[0]
            if "progressiva di fine strada" in group.columns
            else "N/D"
        )

        # Costruiamo delle stringhe descrittive per i Popup sulla mappa
        start_labels.append(
            f"<b>INIZIO:</b> {loc_inizio} (Km {km_inizio})<br><b>Strada:</b> {cod_servizio} {denominazione}"
        )
        end_labels.append(
            f"<b>FINE:</b> {loc_fine} (Km {km_fine})<br><b>Strada:</b> {cod_servizio} {denominazione}"
        )

    # Creazione dei GeoDataFrame 
    gdf_start = gpd.GeoDataFrame(geometry=starts, crs=gdf_spatial.crs)
    gdf_start["Info"] = start_labels

    gdf_end = gpd.GeoDataFrame(geometry=ends, crs=gdf_spatial.crs)
    gdf_end["Info"] = end_labels

    # Generazione della mappa interattiva 
    mappa = gdf_spatial.explore(
        color="black", lwd=3, tiles="OpenStreetMap", name="Rete Stradale"
    )

    mappa = gdf_start.explore(
        m=mappa,
        color="green",
        marker_kwds=dict(radius=7, fill=True),
        # column="Info", 
        name="Punti di Inizio (Km 0)",
        tooltip="Info",
        popup=True,
    )

    mappa = gdf_end.explore(
        m=mappa,
        color="red",
        marker_kwds=dict(radius=7, fill=True),
        # column="Info",
        name="Punti di Fine",
        tooltip="Info",
        popup=True,
    )

    return mappa

def compute_start_end(subdf):
    """Calcola l'inizio e la fine geometrica seguendo il flusso
    dei segmenti (da Km minori a Km maggiori).
    """
    merged = linemerge(union_all(subdf.geometry))

    if merged.geom_type == "LineString":
        endpoints = [Point(merged.coords[0]), Point(merged.coords[-1])]
    elif merged.geom_type == "MultiLineString":
        temp_endpoints = [Point(line.coords[0]) for line in merged.geoms] + [
            Point(line.coords[-1]) for line in merged.geoms
        ]

        max_dist = -1
        p1_max, p2_max = None, None
        for p1, p2 in itertools.combinations(temp_endpoints, 2):
            dist = p1.distance(p2)
            if dist > max_dist:
                max_dist = dist
                p1_max, p2_max = p1, p2
        endpoints = [p1_max, p2_max]

    pt_a, pt_b = endpoints

    net_dx, net_dy = 0, 0
    for geom in subdf.geometry:
        if geom.geom_type == "LineString":
            net_dx += geom.coords[-1][0] - geom.coords[0][0]
            net_dy += geom.coords[-1][1] - geom.coords[0][1]
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                net_dx += line.coords[-1][0] - line.coords[0][0]
                net_dy += line.coords[-1][1] - line.coords[0][1]

    v_ab_x = pt_b.x - pt_a.x
    v_ab_y = pt_b.y - pt_a.y
    dot_product = (v_ab_x * net_dx) + (v_ab_y * net_dy)

    if dot_product > 0:
        return (pt_a.x, pt_a.y), (pt_b.x, pt_b.y)
    else:
        return (pt_b.x, pt_b.y), (pt_a.x, pt_a.y)

## ===== PLOTS ====== 

def plot_map_trentino(geodf, title, ax):
    geodf.plot(
        ax=ax,
        edgecolor='black',
        color = 'white',
        linewidth = 0.4
    )
    ax.set_axis_off()
    plt.title(title, fontsize=20, fontweight='bold', pad=20)

## ===== GENERAL UTILITIES ======    

def create_geodf_from_geojs(geojs):
    gdf= gpd.GeoDataFrame.from_features(geojs["features"])
    gdf.set_crs(epsg=4326, inplace=True)
    gdf['random_values'] = np.random.random(len(gdf))
    return gdf

def top_memory_objects(namespace, n=15):
    """Report the n largest objects in a namespace by size in MiB."""
    sizes = [
        (name, sys.getsizeof(obj) / 1024**2)
        for name, obj in namespace.items()
        if not name.startswith("_")
    ]
    sizes.sort(key=lambda pair: pair[1], reverse=True)
    for name, mib in sizes[:n]:
        print(f"{name:30s} {mib:10.2f} MiB")

    