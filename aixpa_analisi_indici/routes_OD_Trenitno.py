import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

# =============================================================================
# 1. CONFIGURAZIONE
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

CONFIG = {
    "shapefile_path": BASE_DIR /"Com01012026_g/Com01012026_g_WGS84.shp",
    "cod_provincia": "022",  # prefisso ISTAT del Trentino (3 cifre)

    # None = tutta Italia. Per provare/limitare: lista di prefissi provinciali,
    # es. ["021", "025", "024", "023", "017", "014"] (BZ, BL, VI, VR, BS, SO)
    "province_extra": None,

    # --- Cache ---
    "cache_punti_path": BASE_DIR / "capoluoghi_trentino_cache.csv",
    "cache_matrice_path": BASE_DIR / "matrice_tempi_trentino.csv",
    "cache_osrm_dir": BASE_DIR / "cache_osrm",
    "force_recompute": False,

    # --- Server ---
    "user_agent": "overtourism-trentino/0.1",
    "osrm_url": "https://router.project-osrm.org",
    "osrm_max_coord_per_req": 100,  # demo: 100. Con un OSRM locale si può alzare
    "osrm_delay_sec": 1.0,          # con un OSRM locale: 0
    "overpass_urls": [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ],
    "nominatim_url": "https://nominatim.openstreetmap.org/search",
    "nominatim_delay_sec": 1.0,
    "overpass_max_retries": 3,
    "overpass_timeout_sec": 180,

    "test_pairs": [
        ("Trento", "Rovereto"),
        ("Trento", "Riva del Garda"),
        ("Trento", "Pinzolo"),
        ("Trento", "Canazei"),
        ("Trento", "Primiero San Martino di Castrozza"),
        ("Cles", "Cavalese"),
        ("Riva del Garda", "Canazei"),
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# 2. PUNTI DEI COMUNI
# =============================================================================
def init_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session

def cache_path(chiave: str) -> Path:
    p = CONFIG[chiave]
    pe = CONFIG["province_extra"]
    if pe is None:
        italia = p.with_name(f"{p.stem}_italia{p.suffix}")
        return italia if italia.exists() else p
    sfx = "_".join(sorted(pe))
    return p.with_name(f"{p.stem}_{sfx}{p.suffix}")

def carica_comuni() -> gpd.GeoDataFrame:
    """Legge lo shapefile, marca i comuni trentini e (opzionale) filtra le province extra.
    I comuni trentini vengono messi IN CIMA alla lista: serve al calcolo a rettangoli."""
    comuni = gpd.read_file(CONFIG["shapefile_path"], encoding="utf-8").to_crs(epsg=4326)
    comuni = comuni[["PRO_COM_T", "COMUNE", "geometry"]].copy()

    prefisso = comuni["PRO_COM_T"].str[:3]
    comuni["is_trentino"] = prefisso == CONFIG["cod_provincia"]
    if CONFIG["province_extra"] is not None:
        comuni = comuni[comuni["is_trentino"] | prefisso.isin(CONFIG["province_extra"])]

    return comuni.sort_values(["is_trentino", "COMUNE"], ascending=[False, True], ignore_index=True)


def fetch_overpass_capoluoghi(session: requests.Session, trentino: gpd.GeoDataFrame) -> Dict[str, Tuple[float, float, str]]:
    """admin_centre dei comuni trentini (la query riguarda solo il Trentino)."""
    w, s, e, n = trentino.total_bounds
    query = f'''[out:json][timeout:{CONFIG['overpass_timeout_sec']}];
    rel["boundary"="administrative"]["admin_level"="8"]["ref:ISTAT"~"^{CONFIG['cod_provincia']}"]({s},{w},{n},{e});
    foreach(out tags; node(r:"admin_centre"); out;);'''

    for tentativo in range(CONFIG["overpass_max_retries"]):
        for url in CONFIG["overpass_urls"]:
            try:
                r = session.post(url, data={"data": query}, timeout=CONFIG["overpass_timeout_sec"])
                r.raise_for_status()
                data = r.json()
                if "runtime error" in data.get("remark", ""):
                    raise RuntimeError(data["remark"])

                capoluoghi = {}
                istat_corrente = None
                for el in data.get("elements", []):
                    if el["type"] == "relation":
                        istat_corrente = el["tags"].get("ref:ISTAT")
                    elif el["type"] == "node" and istat_corrente:
                        capoluoghi.setdefault(istat_corrente, (el["lon"], el["lat"], el["tags"].get("name")))
                return capoluoghi
            except (requests.RequestException, RuntimeError, KeyError, ValueError) as err:
                logging.warning(f"Overpass errore su {url}: {err}")

        attesa = 15 * (tentativo + 1)
        logging.info(f"Attesa di {attesa}s prima del prossimo tentativo Overpass...")
        time.sleep(attesa)

    raise RuntimeError("Tutti i server Overpass sono irraggiungibili o in errore.")


def fallback_nominatim(session: requests.Session, nome_comune: str) -> Optional[Tuple[float, float, str]]:
    """Solo per comuni TRENTINI senza admin_centre. Restituisce None se non trovato."""
    params = {"q": f"{nome_comune}, Provincia di Trento, Italia", "format": "json", "limit": 1}
    try:
        r = session.get(CONFIG["nominatim_url"], params=params, timeout=30)
        r.raise_for_status()
        res = r.json()
    except (requests.RequestException, ValueError) as err:
        logging.warning(f"Nominatim errore per {nome_comune}: {err}")
        res = []
    finally:
        time.sleep(CONFIG["nominatim_delay_sec"])  # policy: max 1 richiesta al secondo

    if not res:
        return None
    return float(res[0]["lon"]), float(res[0]["lat"]), res[0].get("name", nome_comune)


def get_o_calcola_punti(session: requests.Session) -> pd.DataFrame:
    cache_file = cache_path("cache_punti_path")
    if cache_file.exists() and not CONFIG["force_recompute"]:
        logging.info(f"Carico i punti dalla cache: {cache_file.name}")
        return pd.read_csv(cache_file, dtype={"PRO_COM_T": str})

    comuni = carica_comuni()
    trentino = comuni[comuni["is_trentino"]]
    logging.info(f"{len(comuni)} comuni totali, di cui {len(trentino)} trentini. Interrogo Overpass...")

    cap = fetch_overpass_capoluoghi(session, trentino)
    cap_df = pd.DataFrame.from_dict(cap, orient="index", columns=["lon", "lat", "capoluogo"])
    punti = comuni.merge(cap_df, left_on="PRO_COM_T", right_index=True, how="left").reset_index(drop=True)
    punti["fonte"] = np.where(punti["lon"].notna(), "osm_admin_centre", "")

    # 1) comuni trentini senza admin_centre -> Nominatim
    mancanti_tn = punti.index[punti["is_trentino"] & punti["lon"].isna()]
    logging.info(f"{len(mancanti_tn)} comuni trentini senza admin_centre: provo Nominatim")
    for i in mancanti_tn:
        ris = fallback_nominatim(session, punti.at[i, "COMUNE"])
        if ris is not None:
            punti.at[i, "lon"], punti.at[i, "lat"], punti.at[i, "capoluogo"] = ris
            punti.at[i, "fonte"] = "nominatim"

    # 2) tutti gli altri (fuori provincia, o Nominatim fallito) -> punto interno al poligono
    resto = punti["lon"].isna()
    rp = punti.geometry.representative_point()
    punti.loc[resto, "lon"] = rp.x[resto]
    punti.loc[resto, "lat"] = rp.y[resto]
    punti.loc[resto, "capoluogo"] = punti.loc[resto, "COMUNE"]
    punti.loc[resto, "fonte"] = "punto_interno"

    assert punti["lon"].notna().all() and punti["lat"].notna().all()

    out = pd.DataFrame(punti[["PRO_COM_T", "COMUNE", "capoluogo", "lon", "lat", "is_trentino", "fonte"]])
    out.to_csv(cache_file, index=False)
    logging.info(f"Punti salvati in {cache_file}")
    return out


# =============================================================================
# 3. ROUTING CON OSRM
# =============================================================================
def fetch_osrm_table(session, origini, destinazioni, tentativi: int = 5) -> dict:
    coords = ";".join(f"{lon},{lat}" for lon, lat in origini + destinazioni)
    url = f"{CONFIG['osrm_url']}/table/v1/driving/{coords}"
    params = {
        "sources": ";".join(str(i) for i in range(len(origini))),
        "destinations": ";".join(str(i) for i in range(len(origini), len(origini) + len(destinazioni))),
        "annotations": "duration,distance",
    }
    errore = ""
    for t in range(tentativi):
        try:
            r = session.get(url, params=params, timeout=120)
        except requests.RequestException as err:
            errore = str(err)
        else:
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()  # altri errori 4xx: definitivi, inutile ritentare
                js = r.json()
                if js.get("code") != "Ok":
                    raise RuntimeError(f"OSRM: {js.get('code')} {js.get('message', '')}")
                return js
            errore = f"HTTP {r.status_code}"
        attesa = 5 * 2 ** t
        logging.warning(f"OSRM {errore}: riprovo tra {attesa}s")
        time.sleep(attesa)
    raise RuntimeError(f"OSRM non risponde: {errore}")


def blocco_osrm(session, coords, o, d, cache_dir: Path, etichetta: str):
    """Calcola un blocco (o = indici origini, d = indici destinazioni), con cache su disco."""
    f = cache_dir / f"{etichetta}_{o[0]}_{d[0]}_{len(o)}_{len(d)}.npz"
    if f.exists():
        z = np.load(f)
        return z["durate"], z["distanze"], z["aggancio"]

    resp = fetch_osrm_table(session, [coords[i] for i in o], [coords[j] for j in d])
    durate = np.array(resp["durations"], dtype=float)    # secondi; None -> NaN
    distanze = np.array(resp["distances"], dtype=float)  # metri
    aggancio = np.array([s["distance"] for s in resp["sources"]], dtype=float)
    np.savez(f, durate=durate, distanze=distanze, aggancio=aggancio)
    time.sleep(CONFIG["osrm_delay_sec"])
    return durate, distanze, aggancio


def calcola_matrice_osrm(session, punti_df: pd.DataFrame):
    """Calcola solo le coppie con origine O destinazione in Trentino, a due rettangoli:
       A: origini Trentino  x  destinazioni tutti
       B: origini fuori     x  destinazioni Trentino
    (i comuni trentini sono i primi nT della lista)."""
    coords = list(zip(punti_df["lon"], punti_df["lat"]))
    is_tn = punti_df["is_trentino"].to_numpy()
    N, nT = len(coords), int(is_tn.sum())
    assert is_tn[:nT].all() and not is_tn[nT:].any(), "i comuni trentini devono essere i primi"

    passo = CONFIG["osrm_max_coord_per_req"] // 2  # origini + destinazioni <= limite

    def chunks(idx):
        return [] if len(idx) == 0 else np.array_split(idx, int(np.ceil(len(idx) / passo)))

    idx_T, idx_X, idx_all = np.arange(nT), np.arange(nT, N), np.arange(N)

    # la cache dipende dai punti: se cambiano, cambia la cartella
    tag = hashlib.md5(punti_df[["lon", "lat"]].round(6).to_csv(index=False).encode()).hexdigest()[:8]
    cache_dir = CONFIG["cache_osrm_dir"] / f"{tag}_{passo}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    lavori = [("A", o, d) for o in chunks(idx_T) for d in chunks(idx_all)] + \
             [("B", o, d) for o in chunks(idx_X) for d in chunks(idx_T)]
    cache_files = [
        cache_dir / f"{rett}_{o[0]}_{d[0]}_{len(o)}_{len(d)}.npz"
        for rett, o, d in lavori
    ]
    cache_hits = sum(f.exists() for f in cache_files)
    logging.info(
        f"Blocchi OSRM: {len(lavori)} totali, {cache_hits} in cache, "
        f"{len(lavori) - cache_hits} da richiedere"
    )

    durA, disA = np.full((nT, N), np.nan), np.full((nT, N), np.nan)
    durB, disB = np.full((N - nT, nT), np.nan), np.full((N - nT, nT), np.nan)
    aggancio = np.full(N, np.nan)

    t0 = time.time()
    for k, (rett, o, d) in enumerate(lavori, start=1):
        dur, dis, agg = blocco_osrm(session, coords, o, d, cache_dir, rett)
        if rett == "A":
            durA[np.ix_(o, d)], disA[np.ix_(o, d)] = dur, dis
        else:
            durB[np.ix_(o - nT, d)], disB[np.ix_(o - nT, d)] = dur, dis
        aggancio[o] = agg
        if k % 25 == 0 or k == len(lavori):
            logging.info(f"  {k}/{len(lavori)} blocchi ({(time.time() - t0) / 60:.1f} min)")

    return durA, disA, durB, disB, aggancio


def get_o_calcola_matrice(session, punti_df: pd.DataFrame) -> pd.DataFrame:
    cache_file = cache_path("cache_matrice_path")
    if cache_file.exists() and not CONFIG["force_recompute"]:
        logging.info(f"Carico la matrice dalla cache: {cache_file.name}")
        return pd.read_csv(cache_file, dtype={"istat_origine": str, "istat_destinazione": str})

    durA, disA, durB, disB, aggancio = calcola_matrice_osrm(session, punti_df)
    nT, N = durA.shape

    # --- diagnostica ---
    sub = durA[:, :nT]  # blocco Trentino x Trentino
    asim = np.abs(sub - sub.T) / 60
    logging.info(f"Asimmetria andata/ritorno (TN-TN): mediana {np.nanmedian(asim):.1f} min, max {np.nanmax(asim):.1f} min")
    logging.info(f"Aggancio alla rete: mediana {np.nanmedian(aggancio):.0f} m, "
                 f"max {np.nanmax(aggancio):.0f} m ({punti_df['COMUNE'].iloc[np.nanargmax(aggancio)]})")

    # --- tabella finale: indici di origine/destinazione dei due rettangoli ---
    o = np.concatenate([np.repeat(np.arange(nT), N), np.repeat(np.arange(nT, N), nT)])
    d = np.concatenate([np.tile(np.arange(N), nT), np.tile(np.arange(nT), N - nT)])
    dur = np.concatenate([durA.ravel(), durB.ravel()])
    dis = np.concatenate([disA.ravel(), disB.ravel()])

    nomi = punti_df["COMUNE"].to_numpy()
    istat = punti_df["PRO_COM_T"].to_numpy()
    tot_minuti = dur / 60
    matrice = pd.DataFrame({
        "istat_origine": istat[o],
        "comune_origine": nomi[o],
        "istat_destinazione": istat[d],
        "comune_destinazione": nomi[d],
        "distanza": np.round(dis / 1000, 2),                # km
        "ore": (tot_minuti // 60).astype(int),              # ore intere
        "minuti": np.round(tot_minuti % 60, 1),             # minuti rimanenti
        "tempo_totale_minuti": np.round(tot_minuti, 1)      # (opzionale) comodo se ti serve il totale in minuti per i filtri
    })

    prima = len(matrice)
    matrice = matrice.dropna(subset=["tempo_totale_minuti"])  # es. isole non raggiungibili via strada
    logging.info(f"Righe: {len(matrice)} (scartate {prima - len(matrice)} coppie senza percorso)")

    matrice.to_csv(cache_file, index=False)
    logging.info(f"Matrice salvata in {cache_file}")
    return matrice


# =============================================================================
# 4. CONSULTAZIONE
# =============================================================================
def _codice_istat(punti_df: pd.DataFrame, valore: str) -> str:
    """Accetta un nome di comune o un codice ISTAT; segnala i nomi ambigui."""
    v = str(valore).strip()
    if v in set(punti_df["PRO_COM_T"]):
        return v
    m = punti_df[punti_df["COMUNE"].str.lower() == v.lower()]
    if m.empty:
        raise KeyError(f"Comune '{valore}' non trovato.")
    if len(m) > 1:
        raise KeyError(f"'{valore}' è ambiguo, usa il codice ISTAT: {m['PRO_COM_T'].tolist()}")
    return m["PRO_COM_T"].iloc[0]


def get_tempo_e_distanza(matrice: pd.DataFrame, punti_df: pd.DataFrame, origine: str, destinazione: str) -> str:
    try:
        o, d = _codice_istat(punti_df, origine), _codice_istat(punti_df, destinazione)
    except KeyError as err:
        return str(err.args[0])
    r = matrice[(matrice["istat_origine"] == o) & (matrice["istat_destinazione"] == d)]
    if r.empty:
        return f"Percorso da '{origine}' a '{destinazione}' non presente (nessuno dei due è in Trentino, o senza percorso)."
    riga = r.iloc[0]
    return f"Da {riga['comune_origine']} a {riga['comune_destinazione']}: {riga['tempo_totale_minuti']} min ({riga['distanza']} km)"


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    session = init_session(CONFIG["user_agent"])
    punti_df = get_o_calcola_punti(session)
    matrice = get_o_calcola_matrice(session, punti_df)

    print("\n--- Esempi selezionati ---")
    for o, d in CONFIG["test_pairs"]:
        print(get_tempo_e_distanza(matrice, punti_df, o, d))

    print("\n--- Top 5 percorsi più lunghi ---")
    print(matrice.nlargest(5, "tempo_totale_minuti"))
    return matrice, punti_df


if __name__ == "__main__":
    matrice, punti_df = main()