import plotly.graph_objects as go 
import pandas as pd 


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
    
    return fig 
