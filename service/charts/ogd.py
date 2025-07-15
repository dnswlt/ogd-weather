from concurrent.futures import ThreadPoolExecutor
import os
import requests
import sys
from urllib.parse import urlparse


def fetch_paginated(url, timeout=10):
    """Yields paginated resources from a web API that uses HAL-style links."""
    next_url = url

    while next_url:
        try:
            resp = requests.get(next_url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(f"Failed to fetch or parse {next_url}: {e}") from e

        yield data

        links = data.get("links", [])
        if not isinstance(links, list):
            break
        next_url = next(
            (link.get("href") for link in links if link.get("rel") == "next"), None
        )


def download_csv(url, output_dir, skip_existing=True):
    try:
        filename = os.path.basename(urlparse(url).path)
        output_path = os.path.join(output_dir, filename)

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping existing file {output_path}")
            return

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed: {url} — {e}")


def fetch_all_data(weather_dir: str):
    collections = []
    for r in fetch_paginated("https://data.geo.admin.ch/api/stac/v1/collections"):
        collections.extend(r["collections"])

    print(f"Read {len(collections)} collections.")
    cs = {c["id"]: c for c in collections}

    # Find "items' (data) and "assets" (metadata).
    assets_url = None
    items_url = None
    for l in cs["ch.meteoschweiz.ogd-smn"]["links"]:
        if l["rel"] == "assets":
            assets_url = l["href"]
        elif l["rel"] == "items":
            items_url = l["href"]

    assets = []
    for r in fetch_paginated(assets_url):
        assets.extend(r["assets"])
    print(f"Read {len(assets)} assets.")

    features = []
    for r in fetch_paginated(items_url):
        features.extend(r["features"])
    print(f"Read {len(features)} features.")

    csv_urls = []
    for feature in features:
        for asset in feature["assets"].values():
            if asset["href"].endswith("d_historical.csv"):
                csv_urls.append(asset["href"])

    print(f"Found {len(csv_urls)} CSV URLs. Example: {csv_urls[0]}")

    os.makedirs(weather_dir, exist_ok=True)
    for a in assets:
        download_csv(a["href"], output_dir=weather_dir)

    # Download CSV data files concurrently.
    def _download(url):
        download_csv(url, weather_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_download, csv_urls)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <output_dir>")
        os.exit(1)
    weather_dir = sys.argv[1]
    fetch_all_data(weather_dir=weather_dir)


if __name__ == "__main__":
    main()
