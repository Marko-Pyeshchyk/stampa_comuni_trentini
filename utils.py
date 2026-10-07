# SPDX-License-Identifier: Apache-2.0
from pandas.core.interchange.dataframe_protocol import DataFrame
import digitalhub as dh
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import tempfile
import io
import boto3
import configparser
import json
import logging


PROJECT = os.environ.get("PROJECT_NAME", "overtourism")
DATA_PREFIX = os.environ.get("DATA_PREFIX", "overtourism/inputdata/")
BASE_DIR = os.environ.get("BASE_DIR", str(Path.cwd()))

PATH_SAVE = Path(__file__).resolve().parents[1] / "saved_data"
PATH_SAVE.mkdir(parents=True, exist_ok=True)

## S3 utilities and getter functions from platform

def get_dataframe(name: str) -> DataFrame:
    return dh.get_dataitem(name, project=PROJECT).as_df()

def get_json_s3(name: str) -> dict:
    """Scarica e parsa un JSON da S3, es. get_json_s3('mapping_ids/mapping_comuni_ISTAT.json')."""
    buffer = get_s3(name)
    return json.load(buffer)


def put_dataframe(
    df: pd.DataFrame, name: str, type: str = "parquet", path: Path = PATH_SAVE
) -> str:
    """Saves a dataframe"""
    path.mkdir(parents=True, exist_ok=True)
    path = path / name
    match type:
        case "json":
            path = path.with_suffix(".json")
            df.to_json(path, orient="index", indent=4)
        case "csv":
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)  # added index=False
        case "parquet":
            path = path.with_suffix(".parquet")
            df.to_parquet(path, index=False)  # added index=False
        case _:
            raise NotImplementedError(f"Unsupported type: {type}")
    return str(path)


def log_dataframe(df: pd.DataFrame, name: str, type: str = "parquet"):
    """Uploads a dataframe."""
    file_path = put_dataframe(df, name, type=type)
    project = dh.get_or_create_project(PROJECT)

    if type in {"parquet", "csv"}:
        return project.log_table(f"{name}.{type}", source=file_path, file_format=type)

    raise NotImplementedError(f"Unsupported type: {type}.")



def put_geojson(gdf: gpd.GeoDataFrame, name: str) -> str:
    path = (PATH_SAVE / "overtourism-data" / name).with_suffix(".geojson")
    gdf.to_file(path, driver="GeoJSON")
    return str(path)

def log_geojson(gdf: gpd.GeoDataFrame, name: str) -> None:
    source = put_geojson(gdf, name)
    project = dh.get_or_create_project(PROJECT)
    project.log_artifact(name + ".geojson", "artifact", source=source)

def init_s3_dhcli(env="most-platform"):
    """
    Initialize S3 connection for overtourism analysis.
        Parameters:
            env: environment name
        Returns:
            s3: S3 resource object
            bucket: S3 bucket object
    NOTE: Specific to the DHCLI-based access to platform.
    """
    home = Path.home()

    config = configparser.ConfigParser()
    config.read(home / ".dhcore.ini")
    aws_endpoint_url = config[env]["aws_endpoint_url"]
    aws_access_key_id = config[env]["aws_access_key_id"]
    aws_secret_access_key = config[env]["aws_secret_access_key"]
    aws_session_token = config[env]["aws_session_token"]

    s3 = boto3.resource(
        "s3",
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    bucket = s3.Bucket("most-datalake")
    return s3, bucket

s3, bucket = None, None


def init_s3(force=False):
    global s3, bucket
    if s3 is None or bucket is None or force:
        s3, bucket = init_s3_dhcli("most-platform")
    return s3, bucket

def get_s3(name: str):
    s3, _ = init_s3()
    object = s3.Object("most-datalake", "overtourism/inputdata/" + name)

    buffer = io.BytesIO()
    object.download_fileobj(buffer)
    buffer.seek(0)
    return buffer


def read_shapefile_s3(base_path: str) -> gpd.GeoDataFrame:
    """
    Downloads shapefile files (.shp, .shx, .dbf, .prj) and returns a pandas GeoDataFrame.
    """
    _, bucket = init_s3()

    if base_path.endswith(".shp"):
        base_path = base_path[:-4]

    s3_prefix = "overtourism/inputdata/" + base_path

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downloaded_shp = None

        for obj in bucket.objects.filter(Prefix=s3_prefix):
            file_name = Path(obj.key).name
            local_file_path = temp_path / file_name

            bucket.download_file(obj.key, str(local_file_path))

            if file_name.endswith(".shp"):
                downloaded_shp = local_file_path

        if downloaded_shp is None:
            raise FileNotFoundError(
                f"Nessun file .shp trovato per il prefisso: {s3_prefix}"
            )

        gdf = gpd.read_file(downloaded_shp)
    return gdf


def get_mapping(mapping_name):
    return get_json_s3(f"mapping_ids/{mapping_name}")


def save_computed_dfs(dict_dfs, local=False, type_format = 'parquet', path_saving = PATH_SAVE):
    assert type_format in ["csv", "parquet"]
    for key, value in dict_dfs.items():
        logging.info(f"Uploading dataframe '{key}' in path {path_saving}/{key}.{type_format}...")
        put_dataframe(value, key, type=type_format, path=path_saving)
        if not local:
            logging.info(f"Logging dataframe '{key}.{type_format}'...")
            log_dataframe(value, key, type=type_format)
    logging.info("## Saved.")


if __name__ == "__main__":
    df = get_dataframe("popolazione_2020_2024")
    print(df.head())
