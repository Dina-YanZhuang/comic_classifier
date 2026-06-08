import pandas as pd
import os
import time
import requests
import multiprocessing
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import glob
from PIL import Image

with open('/home/tmpuser/YanLinkopingUni/kblabb-examples/pw.txt', 'r') as file:
    pw = file.read().strip()
AUTH = HTTPBasicAuth('demo', pw)

df = pd.read_feather("/media/tmpuser/DATA/yan_serier/df_content.feather")
df["filename"] = (
    df["dark_id"]
    + "_part"
    + df["part"].str.pad(width=2, side="left", fillchar="0")
    + "_page"
    + df["page"].str.pad(width=3, side="left", fillchar="0")
    + ".jpg"
)

df_page = df.drop_duplicates(subset=["filename"], keep="first")


def download_page_image(
    page_image_url,
    filename,
    backoff_factor=0.1,
    output_height=1600,
    output_folder="/media/tmpuser/DATA/yan_serier/DNserier_test",
):

    os.makedirs(output_folder, exist_ok=True)

    for i in range(5):
        backoff_time = backoff_factor * (2 ** i)

        response = requests.get(
            url=f"{page_image_url}/full/,{output_height}/0/default.jpg",
            auth=AUTH,
        )

        if response.status_code == 200:
            image = response.content
            with open(f"{output_folder}/{filename}", "wb") as image_file:
                image_file.write(image)
            return

        time.sleep(backoff_time)

    print(f"Failed to download {filename} after 5 retries")


backoff_factor = 0.02

url_and_filename = list(df_page[["page_image_url", "filename"]].itertuples(index=False))
args = [(x[0], x[1], backoff_factor) for x in url_and_filename]

pool = multiprocessing.Pool()
pool.starmap(download_page_image, tqdm(args), chunksize=20)
pool.close()


def get_local_image_size(df, image_folder="/media/tmpuser/DATA/yan_serier/DNserier_test"):
    local_image_sizes = []

    for filename, page_id in zip(df.filename, df.page_id):
        try:
            image = Image.open(f"{image_folder}/{filename}")
            width, height = image.size
            local_image_sizes.append(
                {
                    "page_id": page_id,
                    "page_image_width_local": width,
                    "page_image_height_local": height,
                }
            )
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
            local_image_sizes.append(
                {
                    "page_id": page_id,
                    "page_image_width_local": None,
                    "page_image_height_local": None,
                }
            )

    df_local_image_size = pd.DataFrame(local_image_sizes)
    df = pd.merge(df, df_local_image_size, how="left", on="page_id")

    return df


df_page = get_local_image_size(df_page)
df = pd.merge(
    df,
    df_page[["page_id", "page_image_width_local", "page_image_height_local"]],
    how="left",
    on="page_id",
)

df.to_feather("/media/tmpuser/DATA/yan_serier/df_content.feather")
