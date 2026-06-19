import geopandas as gpd 
import numpy as np 
import matplotlib.pyplot as plt 
from shapely.ops import unary_union, linemerge
import sys 

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

def plot_road_direction(gdf, ax):
    for _, group in gdf.groupby('str_cd'):
        merged = linemerge(unary_union(group.geometry))
        
        if merged.geom_type == 'MultiLineString':
            lines = sorted(merged.geoms, key=lambda l: l.coords[0][0])
            x_start, y_start = lines[0].coords[0]
            x_end, y_end = lines[-1].coords[-1]
        else:
            x_start, y_start = merged.coords[0]
            x_end, y_end = merged.coords[-1]
        
        ax.plot(x_start, y_start, marker='o', color='green', markersize=10, zorder=3)
        ax.plot(x_end, y_end, marker='o', color='red', markersize=10, zorder=3)
        ax.annotate(
            group['cod_ser'].iloc[0],
            xy=(x_start, y_start),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8, fontweight='bold', color='darkslategray',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8),
            zorder=4
        )


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

    