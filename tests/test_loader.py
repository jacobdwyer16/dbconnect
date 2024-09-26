from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import sqlalchemy
from typeguard import TypeCheckError

from dbconnect.csv_engine import CSVEngine
from dbconnect.database_engine import DatabaseEngine

PATH = "testpath"


@pytest.fixture(scope="module")
def db_engine() -> DatabaseEngine:
    return DatabaseEngine()


@pytest.fixture(scope="module")
def csv_engine() -> CSVEngine:
    return CSVEngine()


def test_execute_query_returns_polars_dataframe(db_engine: DatabaseEngine) -> None:
    df_pl: pl.DataFrame = db_engine.execute_query_from_file("test_query.sql")
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The Polars DataFrame should not be empty"


def test_execute_query_returns_pandas_dataframe(db_engine: DatabaseEngine) -> None:
    df_pd: pd.DataFrame = db_engine.execute_query_from_file(
        "test_query.sql"
    ).to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "The result should be a Pandas DataFrame"
    assert not df_pd.empty, "The Pandas DataFrame should not be empty"


def test_execute_csv_engine_pandas(portara_engine: CSVEngine) -> None:
    df_pd: pd.DataFrame = csv_engine.get_df(PATH).to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "The result should be a Pandas DataFrame"
    assert not df_pd.empty, "The Pandas DataFrame should not be empty"


def test_execute_csv_engine_polars(portara_engine: CSVEngine) -> None:
    df_pl: pl.DataFrame = csv_engine.get_df(PATH)
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The polars DataFrame should not be empty"


def test_execute_csv_engine_polars_error() -> None:
    with pytest.raises(ValueError):
        CSVEngine().get_df("JJJJ")


def test_execute_csv_engine_pandas_error() -> None:
    with pytest.raises(ValueError):
        CSVEngine().get_df("ZSA")


def test_load_df_caching(csv_engine: CSVEngine) -> None:
    """Test that _load_df method caches the results."""
    # Load the DataFrame once
    df1 = csv_engine.get_df(PATH)

    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    df2 = csv_engine.get_df(PATH)

    assert df1 is df2, "The DataFrame should be cached and return the same object"


def test_validate_commodity_valid(csv_engine: CSVEngine) -> None:
    """Test that a valid commodity passes validation."""
    try:
        csv_engine.get_df(PATH).to_pandas()
    except ValueError:
        pytest.fail("Valid commodity 'NGA' raised ValueError unexpectedly!")


def test_validate_commodity_invalid_type(csv_engine: CSVEngine) -> None:
    """Test that passing a non-string commodity raises a TypeError due to typeguard."""
    with pytest.raises(TypeCheckError):
        csv_engine.get_df(123).to_pandas()  # Passing integer instead of string


def test_path_setters(csv_engine: CSVEngine) -> None:
    """Test setting new paths using setters."""
    new_expiry_path = "C:/new_expiry_path"
    csv_engine.path = Path(new_expiry_path)
    assert csv_engine.path == new_expiry_path, "Expiry path was not set correctly"


def test_set_invalid_path_type(csv_engine: CSVEngine) -> None:
    """Test that setting paths with invalid types raises a TypeError."""
    with pytest.raises(TypeCheckError):
        csv_engine.path = 12345  # Invalid type

    with pytest.raises(TypeCheckError):
        csv_engine.path = None  # Invalid type

    with pytest.raises(TypeCheckError):
        csv_engine.path = ["invalid", "path"]  # Invalid type


def test_clear_cache_db_engine(db_engine: DatabaseEngine) -> None:
    """Test that the cache is cleared in DatabaseEngine."""
    # Load the DataFrame once to populate the cache
    df1 = db_engine.execute_query_from_file("test_query.sql")

    # Clear the cache
    db_engine.clear_all_caches()

    # Load the DataFrame again
    df2 = db_engine.execute_query_from_file("test_query.sql")

    assert df1 is not df2, "The cache should be cleared and return a new object"


def test_clear_cache_csv_engine(portara_engine: CSVEngine) -> None:
    """Test that the cache is cleared in CSVEngine."""
    # Load the DataFrame once to populate the cache
    df1 = csv_engine.get_df(PATH)

    # Clear the cache
    csv_engine.clear_all_caches()

    # Load the DataFrame again
    df2 = csv_engine.get_df(PATH)
    assert df1 is not df2, "The cache should be cleared and return a new object"


def test_set_empty_current_path(csv_engine: CSVEngine) -> None:
    """Test setting the current path to an empty string and ensure it works."""
    csv_engine.path = ""
    try:
        df = csv_engine.get_df(PATH).to_pandas()
        assert isinstance(df, pd.DataFrame), "Expected a Pandas DataFrame"
    except Exception as e:
        pytest.fail(f"Setting current path to empty string caused an error: {e}")


def test_query_timeout() -> None:
    """Test that setting the timeout to 1 second causes the query to timeout."""
    tester = DatabaseEngine()
    tester.clear_all_caches()
    tester.timeout = 2
    with pytest.raises(sqlalchemy.exc.OperationalError):
        tester.execute_query_from_file("test_query.sql")
