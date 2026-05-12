from typing import Optional, Union

import pandas as pd


def load_csv(
    file: Union[str, "pd.io.common.FilePathOrBuffer"],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file, nrows=nrows)
