import plotly.graph_objects as go 
import pandas as pd 
from scipy.signal import find_peaks

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

def plot_stats(
    df, 
    col_data='date', 
    col_comune='comune', 
    col_valore='value',
    stats=['mean', 'median'],
    peaks_df=None          
):
    """
    Mostra i singoli comuni + le statistiche aggregate selezionate.
    """
    df = df.copy()
    df[col_data] = pd.to_datetime(df[col_data])
    
    stats_df = df.groupby(col_data)[col_valore].agg(stats).reset_index().sort_values(col_data)
    
    fig = go.Figure()
    
    comuni = df[col_comune].unique()
    for i, comune in enumerate(comuni):
        df_comune = df[df[col_comune] == comune].sort_values(col_data)

        fig.add_trace(go.Scatter(
            x=df_comune[col_data],
            y=df_comune[col_valore],
            mode='lines',
            name='Singoli Comuni',
            legendgroup='group_comuni',
            showlegend=(i == 0),
            line=dict(color="#bcccdb", width=1),
            hovertemplate=f"<b>{comune}</b><br>Presenze: %{{y:.2f}}<extra></extra>" 
        ))
        
    style_map = {
        'mean':   dict(name='Media', color='#D62728', width=3, dash='solid'),   
        'median': dict(name='Mediana', color='#2CA02C', width=2.5, dash='solid'), 
        'max':    dict(name='Massimo', color='#FF7F0E', width=2, dash='dot'),     
        'min':    dict(name='Minimo', color='#9467BD', width=2, dash='dot'),      
        'std':    dict(name='Dev. Std', color='#8C564B', width=1.5, dash='dash')
    }

    for stat in stats:
        if stat in stats_df.columns:
            config = style_map.get(stat, dict(name=stat.capitalize(), color='black', width=2, dash='solid'))
            fig.add_trace(go.Scatter(
                x=stats_df[col_data],
                y=stats_df[stat],
                mode='lines',
                name=config['name'],
                line=dict(color=config['color'], width=config['width'], dash=config['dash']),
                hovertemplate=f"<b>{config['name']}</b><br>Valore: %{{y:.2f}}<extra></extra>"
            ))

    if peaks_df is not None:
        fig.add_trace(go.Scatter(
            x=peaks_df[col_data],
            y=peaks_df[col_valore],
            mode='markers',
            name='Picchi',
            marker=dict(
                size=7,
                color='#1F77B4',               
                symbol='circle',
                line=dict(color='white', width=1.5)   #
            ),
            hovertemplate="<b>Picco</b><br>Data: %{x|%Y-%m-%d}<br>Valore: %{y:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Andamento Presenze: Distribuzione vs Statistiche",
        xaxis_title="Data",
        yaxis_title="Presenze Turistiche",
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig



def find_peaks_dynamic(ts, dist = 14, prominence_perc=0.15):
    y_smoothed = ts.dropna() # Rimuoviamo i NaN generati dal rolling

    # 'distance=14' impone che ci siano almeno 7 giorni tra un picco e l'altro
    dynamic_prominence = (y_smoothed.max() - y_smoothed.min()) * prominence_perc
    peaks_indices, _ = find_peaks(y_smoothed, distance=dist, prominence=dynamic_prominence)
    
    dates_peaks = y_smoothed.index[peaks_indices]
    valori_picchi = y_smoothed.iloc[peaks_indices]

    return dates_peaks, valori_picchi, dynamic_prominence

