import geopandas as gpd 
import numpy as np 
import matplotlib.pyplot as plt 
from shapely.ops import linemerge
from shapely import union_all
import sys 
from shapely.geometry import LineString, MultiLineString, Point
import itertools 
import pandas as pd 

## ===== SPECIFIC FUNCTIONS FOR TRAFFICO VEICOLARE ======

def orient_geometry_by_points(geometries, pt_start):
    """Prende una collezione di geometrie (LineString/MultiLineString) e le riordina

    e orienta rigidamente affinché scorrano da pt_start a pt_end.
    """
    # Appiattiamo tutte le geometrie in una lista di LineString semplici
    segments = []
    for geom in geometries:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            segments.append(geom)
        elif geom.geom_type == "MultiLineString":
            segments.extend(geom.geoms)

    if not segments:
        return None

    ordered_segments = []
    current_point = pt_start
    remaining = list(segments)

    # Concatena i segmenti dal punto di inizio a quello di fine
    while remaining:
        best_idx = -1
        best_dist = float("inf")
        flip_needed = False

        for i, seg in enumerate(remaining):
            p_first = Point(seg.coords[0])
            p_last = Point(seg.coords[-1])

            d_first = current_point.distance(p_first)
            d_last = current_point.distance(p_last)

            if d_first < best_dist:
                best_dist = d_first
                best_idx = i
                flip_needed = False
            if d_last < best_dist:
                best_dist = d_last
                best_idx = i
                flip_needed = True

        if best_idx == -1:
            break

        seg = remaining.pop(best_idx)
        if flip_needed:
            seg = LineString(list(seg.coords)[::-1])

        ordered_segments.append(seg)
        current_point = Point(seg.coords[-1])

    try:
        merged = linemerge(ordered_segments)
        if merged.geom_type == "LineString":
            return merged
        else:
            return MultiLineString(ordered_segments)
    except Exception:
        return MultiLineString(ordered_segments)


def create_complete_map(gdf_spatial, df_tabular, df_spire):
    gdf_spatial = gdf_spatial.copy()
    df_tabular = df_tabular.copy()
    df_spire = df_spire.copy()

    # Normalizzazione codici
    gdf_spatial["str_cd"] = gdf_spatial["str_cd"].astype(str).str.strip()
    df_tabular["id_lrs"] = df_tabular["id_lrs"].astype(str).str.strip()
    df_spire["ID_strada"] = df_spire["ID_strada"].astype(str).str.strip()
    df_spire["STRADA"] = df_spire["STRADA"].astype(str).str.strip()

    gdf_merged = gdf_spatial.merge(
        df_tabular, left_on="str_cd", right_on="id_lrs", how="left"
    )

    oriented_roads = {}
    starts, ends = [], []
    start_labels, end_labels = [], []

    # Elaborazione delle strade
    for str_cd, group in gdf_merged.groupby("str_cd"):
        start_coords, end_coords = compute_start_end(group)

        pt_start = Point(start_coords)
        pt_end = Point(end_coords)

        starts.append(pt_start)
        ends.append(pt_end)

        # --- APPLICAZIONE NUOVO ORIENTAMENTO ---
        oriented_line = orient_geometry_by_points(
            group.geometry, pt_start
        )

        try:
            km_inizio_strada = float(
                group["progressiva di inizio strada"].iloc[0]
                if "progressiva di inizio strada" in group.columns
                else 0.0
            )
        except (ValueError, TypeError):
            km_inizio_strada = 0.0

        road_info = {"geometry": oriented_line, "km_inizio": km_inizio_strada}

        oriented_roads[str_cd] = road_info

        col_id_tab = next(
            (
                c
                for c in group.columns
                if c.lower().replace("_", " ").strip() in ["id strada", "idstrada"]
            ),
            None,
        )
        if col_id_tab:
            id_strada_val = str(group[col_id_tab].iloc[0]).strip()
            oriented_roads[id_strada_val] = road_info

        if "codice servizio" in group.columns and pd.notna(
            group["codice servizio"].iloc[0]
        ):
            oriented_roads[str(group["codice servizio"].iloc[0]).strip()] = (
                road_info
            )

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

        start_labels.append(
            f"<b>INIZIO:</b> {loc_inizio} (Km {km_inizio})<br><b>Strada:</b> {cod_servizio} {denominazione}"
        )
        end_labels.append(
            f"<b>FINE:</b> {loc_fine} (Km {km_fine})<br><b>Strada:</b> {cod_servizio} {denominazione}"
        )

    gdf_start = gpd.GeoDataFrame(geometry=starts, crs=gdf_spatial.crs)
    gdf_start["Info"] = start_labels

    gdf_end = gpd.GeoDataFrame(geometry=ends, crs=gdf_spatial.crs)
    gdf_end["Info"] = end_labels

    # 3. Interpolazione delle Spire
    spire_points = []
    spire_labels = []
    valid_spire_rows = []

    for idx, row in df_spire.iterrows():
        id_strada = row["ID_strada"]
        nome_strada = str(row["STRADA"]).strip()

        road_data = None
        if id_strada in oriented_roads:
            road_data = oriented_roads[id_strada]
        elif nome_strada in oriented_roads:
            road_data = oriented_roads[nome_strada]

        if road_data is not None:
            geom_strada = road_data["geometry"]
            km_inizio_strada = road_data["km_inizio"]
            km_spira = float(row["KM"])

            dist_km = km_spira - km_inizio_strada
            distanza_metri = dist_km * 1000

            if 0 <= distanza_metri <= geom_strada.length:
                point_interpolated = geom_strada.interpolate(distanza_metri)
                spire_points.append(point_interpolated)
                label = f"<b>Rilevazione:</b> {row['LOCALITA']} (Stazione n. {row['NUMERO']})<br><b>Strada:</b> {row['STRADA']} (Km {km_spira})"
                spire_labels.append(label)
                valid_spire_rows.append(row)
            else:
                if distanza_metri > geom_strada.length and (
                    distanza_metri - geom_strada.length
                ) < 1500:
                    point_interpolated = geom_strada.interpolate(
                        geom_strada.length
                    )
                    spire_points.append(point_interpolated)
                    label = f"<b>Rilevazione:</b> {row['LOCALITA']} (Stazione n. {row['NUMERO']})<br><b>Strada:</b> {row['STRADA']} (Km {km_spira} - Fine geometrica)"
                    spire_labels.append(label)
                    valid_spire_rows.append(row)

    if spire_points:
        gdf_spire_map = gpd.GeoDataFrame(
            pd.DataFrame(valid_spire_rows).reset_index(drop=True),
            geometry=spire_points,
            crs=gdf_spatial.crs,
        )
        gdf_spire_map["Info"] = spire_labels
        print(
            f"Posizionate {len(gdf_spire_map)} spire su {len(df_spire)}."
        )
    else:
        gdf_spire_map = gpd.GeoDataFrame(
            columns=["Info"], geometry=[], crs=gdf_spatial.crs
        )

    # 4. Mappa Folium
    mappa = gdf_spatial.explore(
        color="black", lwd=3, tiles="OpenStreetMap", name="Rete Stradale"
    )
    mappa = gdf_start.explore(
        m=mappa,
        color="green",
        marker_kwds=dict(radius=7, fill=True),
        name="Punti di Inizio",
        tooltip="Info",
        popup=True,
    )
    mappa = gdf_end.explore(
        m=mappa,
        color="red",
        marker_kwds=dict(radius=7, fill=True),
        name="Punti di Fine",
        tooltip="Info",
        popup=True,
    )

    if not gdf_spire_map.empty:
        mappa = gdf_spire_map.explore(
            m=mappa,
            color="blue",
            marker_kwds=dict(radius=6, fill=True),
            name="Punti di Rilevazione (Spire)",
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

    