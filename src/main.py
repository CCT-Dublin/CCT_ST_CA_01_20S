import logging
import polars as pl
import pandas as pd
from pathlib import Path
from utils.helpers import *
from typing import List, Dict


# ******************************************************************************************************************** #
#                                              System Logging                                                          #
# ******************************************************************************************************************** #
logging.basicConfig(
    format=("%(asctime)s | %(levelname)s | File_name ~> %(module)s.py "
            "| Function ~> %(funcName)s | Line ~~> %(lineno)d  ~~>  %(message)s"),
    level=logging.INFO
)


# ******************************************************************************************************************** #
#                                              Auxiliary Function                                                      #
# ******************************************************************************************************************** #
def download_dataset() -> pl.DataFrame:
    data_dir = Path("../data")
    dataset_file = data_dir / "creditcard.csv"

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not dataset_file.exists():

            logging.warning("📥 Dataset not found. Downloading from Kaggle...")

            path = kagglehub.dataset_download(
                "mlg-ulb/creditcardfraud",
                output_dir=str(data_dir)
            )

            logging.info(f"✅ Dataset downloaded to: {path}")

        else:

            logging.info(f"✅ Dataset already exists: {dataset_file}")

        return  pl.read_csv(dataset_file, schema_overrides={"Time": pl.Float64})

    except Exception as e:
        logging.error(f"❌ Failed to download or read the dataset: {e}")


# ******************************************************************************************************************** #
#                                               Main function                                                          #
# ******************************************************************************************************************** #
def main() -> None:
    df = download_dataset()
    print(df)


if __name__ == "__main__":
    main()