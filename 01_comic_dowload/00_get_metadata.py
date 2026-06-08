import os
import multiprocessing

from json import loads

# from time import sleep
import pandas as pd
from tqdm import tqdm
from urllib3.util import Retry
from urllib3 import PoolManager, make_headers
from kblab import Archive
from itertools import product
import random

# Create custom API call to download all metadata files instead of using kblab package
# Retry after failed attempts
def get_metadata(dark_id, headers):
    """Custom API call to download every package's metadata (meta.json) files to filter API content.
    kblab package's .search()-method returns incomplete results, where some some ids are missing
    from search results. Downloading metadata of all id's is the most secure way of ensuring all
    packages are included when filtering for dates or newspaper names.
    Args:
        dark_id (str): URI of package in betalab.kb.se or datalab.kb.se
        headers (list): API authentication details passed as header.

    Returns:
        [dict]: dict with fields of meta.json file.
    """

    http = PoolManager()
    try:
        meta_json = http.request(
            "GET",
            f"https://datalab.kb.se/{dark_id}/meta.json",
            headers=headers,
            retries=Retry(connect=5, read=4, redirect=5, backoff_factor=0.02),
        )

        meta_json = loads(meta_json.data.decode("utf-8"))
        meta_json["dark_id"] = dark_id
        return meta_json

    except Exception as e:
        # Return minimal data for failed requests (will be filtered out in Stage 5)
        return {"dark_id": dark_id}


if __name__ == "__main__":
    print("Stage 1: Loading authentication credentials...")
    with open('/home/tmpuser/YanLinkopingUni/kblabb-examples/pw.txt', 'r') as file:
        pw = file.read().replace('\n', '')
    a = Archive("https://datalab.kb.se", auth=("demo", pw))
    print("✓ Authentication successful")

    print("\nStage 2: Generating search query for years 1923-2025...")
    # Generate search query for years 1923-2025
    years = list(range(1923, 2026))
    year_query = " or ".join(str(year) for year in years)
    search_query = f'label: "DAGENS NYHETER" AND meta.created: ({year_query})'
    print(f"Search query: {search_query[:80]}...")
    
    print("\nStage 3: Searching for DAGENS NYHETER issues...")
    dark_ids = a.search(search_query)
    dark_ids = list(dark_ids)
    print(f"✓ Found {len(dark_ids)} issues")

    print("\nStage 4: Fetching metadata for all issues (this may take a while)...")
    headers = make_headers(basic_auth=f"demo:{pw}")
    
    # Prepare tasks
    task_list = list(product(dark_ids, [headers]))
    
    pool = multiprocessing.Pool()
    df_meta = pool.starmap(
        get_metadata,
        tqdm(task_list, desc="Fetching metadata", unit="issue", total=len(task_list)),
        chunksize=5000
    )
    pool.close()
    print(f"✓ Metadata fetched for {len(df_meta)} items")

    print("\nStage 5: Creating DataFrame and processing metadata...")
    os.makedirs("data", exist_ok=True)
    df_meta = pd.DataFrame(df_meta)
    print(f"DataFrame shape before filtering: {df_meta.shape}")
    print(f"DataFrame columns: {list(df_meta.columns)}")
    
    # Remove failed requests (rows where all columns except dark_id are NaN)
    non_dark_id_cols = [col for col in df_meta.columns if col != 'dark_id']
    initial_count = len(df_meta)
    df_meta = df_meta.dropna(subset=non_dark_id_cols, how='all')
    failed_count = initial_count - len(df_meta)
    print(f"DataFrame shape after filtering failed requests: {df_meta.shape}")
    print(f"✓ Removed {failed_count} rows with incomplete metadata")
    
    # Keep only newspaper name, throw away date
    if 'title' in df_meta.columns:
        df_meta["title"] = df_meta["title"].str.extract(r"(\D*) (\D*)?", expand=False).loc[:, 0]
        print(f"✓ Extracted newspaper titles")
    else:
        print("⚠ 'title' column not found in metadata")
    
    # Extract year from created date
    if 'created' in df_meta.columns:
        df_meta['year'] = pd.to_datetime(df_meta['created'], errors='coerce').dt.year
        df_meta['year'] = df_meta['year'].fillna('failed')
    else:
        df_meta['year'] = 'failed'
    
    print("\nStage 6: Sampling one random issue per year...")
    if 'year' in df_meta.columns:
        initial_count = len(df_meta)
        df_meta = df_meta.groupby('year').apply(lambda x: x.sample(n=1, random_state=None)).reset_index(drop=True)
        print(f"✓ Sampled {len(df_meta)} issues ({initial_count} → {len(df_meta)})")
        print(f"✓ Years covered: {sorted(df_meta['year'].unique())}")
    else:
        print("⚠ 'year' column not found in metadata")
    
    nr_failed = len(df_meta[df_meta["year"] == "failed"])
    print(f"\nStage 7: Checking for failed requests...")
    print(f"A total of {nr_failed} failed requests.")

    # df_meta["year"] = pd.to_numeric(df_meta["year"], errors="coerce")
    # df_meta["year"] = df_meta["year"].astype("Int16")
    # print(df_meta)
    # df_meta = df_meta[~df_meta["issue"].isna()]  # Remove ids that aren't newspapers
    # df_meta = df_meta.reset_index(drop=True)

    print("\nStage 8: Saving results to feather file...")
    df_meta.to_feather("/media/tmpuser/DATA/yan_serier/sampled_editions.feather", compression=None)
    print(f"✓ Successfully saved {len(df_meta)} sampled editions to feather file")
    print("✓ Script completed!")
