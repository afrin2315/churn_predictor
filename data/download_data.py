"""Download the IBM Telco Customer Churn dataset.

The script tries the public raw GitHub URL first. If that fails, it prints a
manual Kaggle fallback link so the CSV can be downloaded by hand.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"


def download_dataset(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Dataset already exists at {destination}")
        return destination

    try:
        with urlopen(DATA_URL, timeout=30) as response:
            if response.status != 200:
                raise HTTPError(DATA_URL, response.status, "Unexpected HTTP status", hdrs=None, fp=None)
        urlretrieve(DATA_URL, destination)
        print(f"Downloaded dataset to {destination}")
        return destination
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        message = (
            "Failed to download the IBM raw CSV.\n"
            f"Reason: {exc}\n\n"
            "Download it manually from Kaggle and save it as:\n"
            f"  {destination}\n\n"
            f"Kaggle link: {KAGGLE_URL}"
        )
        raise RuntimeError(message) from exc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "data" / "Telco-Customer-Churn.csv"
    download_dataset(destination)


if __name__ == "__main__":
    main()
