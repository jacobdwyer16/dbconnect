import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
from packaging import version
from typeguard import typechecked

logger = logging.getLogger(__name__)


class CSVEngine:
    """
    CSVEngine is a singleton class responsible for loading and managing CSV files using the Polars library.
    Attributes:
        _instance (CSVEngine): The singleton instance of the CSVEngine class.
        _lock (threading.Lock): A lock to ensure thread-safe singleton initialization.
        _path (Path): The path to the directory containing CSV files.
        _column_mappings (Optional[Dict[str, Dict[str, pl.DataType]]]): Optional mappings of column names to Polars data types.
    Methods:
        __new__(cls, *args: Any, **kwargs: Any) -> CSVEngine:
            Ensures that only one instance of the class is created (singleton pattern).
        check_polars_version() -> None:
            Checks if the installed Polars version is 1.0.0 or higher. Raises ImportError if the version is too old.
        _initialize(paths: Path, column_mappings: Optional[Dict[str, Dict[str, pl.DataType]]] = None) -> None:
            Initializes the CSVEngine instance with the given paths and column mappings.
        path() -> Path:
            Gets the current path to the directory containing CSV files.
        path(value: Path) -> None:
            Sets a new path to the directory containing CSV files.
        _load_file_df(path: Path, column_mapping: Dict[str, pl.DataType] | None) -> pl.DataFrame:
            Loads a CSV file into a Polars DataFrame, applying optional column mappings.
        _load_df() -> pl.DataFrame:
            Loads the DataFrame from the CSV file, using caching to improve performance.
        get_df() -> pl.DataFrame:
            Returns the loaded DataFrame.
        clear_all_caches() -> None:
            Clears all cached data and reinitializes the CSVEngine instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "CSVEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    @staticmethod
    def check_polars_version() -> None:
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            raise ImportError(
                f"Polars version {pl.__version__} is too old."
                "Please upgrade to a version 1.0.0 or higher."
            )

    def _initialize(
        self,
        paths: Path,
        column_mappings: Optional[Dict[str, Dict[str, pl.DataType]]] = None,
    ) -> None:
        self._path = paths
        self._column_mappings = column_mappings

        self.check_polars_version()

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    @typechecked
    def path(self, value: Path) -> None:
        self._path = value

    @staticmethod
    def _load_file_df(
        path: Path, column_mapping: Dict[str, pl.DataType] | None
    ) -> pl.DataFrame:
        if len(str(path)) == 0:
            return pl.DataFrame()

        full_path = str(Path(path)/"*.csv")

        if not Path(path).exists():
            logging.warning(f"Current path {path} does not exist.")
            raise FileNotFoundError(f"Current path {path} does not exist.")

        df: pl.DataFrame = pl.read_csv(
            full_path, infer_schema=False, raise_if_empty=False
        )

        if column_mapping:
            for col, dtype in column_mapping.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(dtype))

        return df

    @lru_cache
    def _load_df(self) -> pl.DataFrame:
        df = self._load_file_df(
            path=self.path,
            column_mapping=self._column_mappings,
        )

        if df.is_empty():
            df = pl.DataFrame()

        return df

    @typechecked
    def get_df(self) -> pl.DataFrame:
        return self._load_df()

    def clear_all_caches(self) -> None:
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "cache_clear"):
                attr.cache_clear()
        self._initialize(
            paths=self._path,
            column_mappings=self._column_mappings,
        )
