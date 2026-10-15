import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_PATH = REPO_ROOT / 'raw_data'

VODAFONE_ATTENDENCIES_PATH = DATA_PATH / 'vodafone_attendences.csv'
VODAFONE_ATTENDENCIES_NEW_PATH = DATA_PATH / 'vodafone_attendences_new.csv'
GEOJSON_COMUNI_VODAFONE_PATH = DATA_PATH / 'TRENTINO-comuni_Vodafone_2023.geojson'

MAPPING_COMUNI_VODAFONE_PATH = DATA_PATH / 'mapping_comuni_into_vodafone_Trento.json'
MAPPING_COMUNI_APT_PATH = DATA_PATH / 'map_comuni_into_apt.json'


s3_files = [
    VODAFONE_ATTENDENCIES_PATH,
    VODAFONE_ATTENDENCIES_NEW_PATH,
    GEOJSON_COMUNI_VODAFONE_PATH,
]

map_files = [
    MAPPING_COMUNI_VODAFONE_PATH,
    MAPPING_COMUNI_APT_PATH,
]


def main():
    # imported here so the notebooks can import the paths above without needing S3 access
    from utils import get_mapping, get_s3

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    for path in s3_files:
        print(f"Downloading {path.name}...")
        buffer = get_s3(path.name)
        path.write_bytes(buffer.getvalue())

    for path in map_files:
        print(f"Downloading mapping {path.name}...")
        mapping = get_mapping(path.name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(s3_files) + len(map_files)} files in {DATA_PATH}")


if __name__ == '__main__':
    main()
