import hashlib
import pandas as pd
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class NamedDataframe:
    names: set[str]
    df: pd.DataFrame

    def add_name(self, name: str):
        if name in self.names:
            print(
                f"{name} is already in NamedDataframe object's set of names:"
                f" {self.names}"
            )
        else:
            self.names.add(name)

    def remove_name(self, name: str):
        if name not in self.names:
            print(
                f"{name} is not in NamedDataframe object's set of names:"
                f" {self.names}"
            )
        elif len(self.names) == 1:
            print(
                f"{name} is the only remaining name of NamedDataframe object."
                " If caller wants to remove object and name, must delete"
                " object."
            )
        else:
            self.names.remove(name)


class DataframeHashVals(NamedTuple):
    content_hash: str
    dtype_hash: str


@dataclass
class DataframeWithHashVals:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def _hash(item: str):
        return hashlib.sha256(item.encode("utf-8")).hexdigest()

    # Dataframe can be modified, so calc hash_vals on each access
    @property
    def hash_vals(self) -> DataframeHashVals:
        return DataframeHashVals(
            content_hash=self._hash(item=self.df.to_csv()),
            dtype_hash=self._hash(item=str(self.df.dtypes)),
        )


class DataFrameCache:
    def __init__(
        self, stored_dataframes: dict[str, DataframeWithHashVals] = None
    ):
        if stored_dataframes is None:
            stored_dataframes = {}
        self._stored_dataframes = stored_dataframes

    @property
    def stored_dataframes_by_name(self) -> dict[str, DataframeHashVals]:
        return {
            name: df.hash_vals for name, df in self._stored_dataframes.items()
        }

    @property
    def stored_dataframes_by_hash_vals(self) -> dict[DataframeHashVals, str]:
        return {
            hash_vals: name
            for name, hash_vals in self.stored_dataframes_by_name.items()
        }

    def add_df(self, df: pd.DataFrame, label: str):
        # TODO exception for case where already have dict entry w/ label
        # TODO ...cont'd: Let Python raise standard dict exception for now.
        df_with_hashes = DataframeWithHashVals(df=df)
        hash_vals = df_with_hashes.hash_vals
        if hash_vals in self.stored_dataframes_by_hash_vals:
            print(
                "Item named"
                f" {self.stored_dataframes_by_hash_vals[hash_vals]} with same"
                " hash vals already saved in cache. Will go ahead and save"
                f" another copy named {label}"
            )
        self._stored_dataframes[label] = DataframeWithHashVals(
            df=df.copy(deep=True)
        )

    def remove_df(self, label: str):
        if label not in self._stored_dataframes:
            print(f"No dataframe named {label} in cache")
        else:
            del self._stored_dataframes[label]

    def get_df(self, label: str) -> pd.DataFrame:
        # TODO custom exception for case when key not found
        return self._stored_dataframes[label].df.copy(deep=True)