import time

import pandas as pd
import polars as pl
import pytest
import sqlalchemy
from typeguard import TypeCheckError

from dbconnect.contract_engine import CSVEngine
from dbconnect.database_engine import DatabaseEngine


@pytest.fixture(scope="module")
def db_engine() -> DatabaseEngine:
    return DatabaseEngine()


@pytest.fixture(scope="module")
def csv_engine() -> CSVEngine:
    return CSVEngine()


def test_execute_query_from_file_returns_polars_dataframe(
    db_engine: DatabaseEngine,
) -> None:
    df_pl: pl.DataFrame = db_engine.execute_query_from_file("test_query.sql")
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The Polars DataFrame should not be empty"


def test_execute_query_returns_polars_dataframe(db_engine: DatabaseEngine) -> None:
    query = db_engine.load_query("test_query.sql")
    df_pl: pl.DataFrame = db_engine.execute_query(query)
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The Polars DataFrame should not be empty"


def test_execute_query_returns_pandas_dataframe(db_engine: DatabaseEngine) -> None:
    df_pd: pd.DataFrame = db_engine.execute_query_from_file(
        "test_query.sql"
    ).to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "The result should be a Pandas DataFrame"
    assert not df_pd.empty, "The Pandas DataFrame should not be empty"


def test_execute_csv_engine_pandas(csv_engine: CSVEngine) -> None:
    df_pd: pd.DataFrame = csv_engine.get_prices_df("TestDoc").to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "The result should be a Pandas DataFrame"
    assert not df_pd.empty, "The Pandas DataFrame should not be empty"


def test_execute_csv_engine_pandas_expiry(csv_engine: CSVEngine) -> None:
    df_pd: pd.DataFrame = csv_engine.get_expiry_df("TestDoc").to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "The result should be a Pandas DataFrame"
    assert not df_pd.empty, "The Pandas DataFrame should not be empty"


def test_execute_csv_engine_polars(csv_engine: CSVEngine) -> None:
    df_pl: pl.DataFrame = csv_engine.get_prices_df("TestDoc")
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The polars DataFrame should not be empty"


def test_execute_csv_engine_polars_expiry(csv_engine: CSVEngine) -> None:
    df_pl: pl.DataFrame = csv_engine.get_expiry_df("TestDoc")
    assert isinstance(df_pl, pl.DataFrame), "The result should be a Polars DataFrame"
    assert not df_pl.is_empty(), "The polars DataFrame should not be empty"


def test_one_date_per_dataframe_prices(csv_engine: CSVEngine) -> None:
    df_pl: pl.DataFrame = csv_engine.get_prices_df("TestDoc")
    dates_unique = df_pl.select(pl.col("Date"), pl.col("Contract")).unique()
    dates = df_pl.select(pl.col("Date"), pl.col("Contract"))
    assert len(dates) == len(dates_unique), "There should not be duplicate dates"


def test_prices_vs_expirty(csv_engine: CSVEngine) -> None:
    df_pl: pl.DataFrame = csv_engine.get_prices_df("TestDoc")
    df_exp: pl.DataFrame = csv_engine.get_expiry_df("TestDoc")
    assert df_pl.height != df_exp.height, "Expiry and Prices df should be different"

def test_execute_csv_engine_polars_error() -> None:
    with pytest.raises(ValueError):
        CSVEngine().get_expiry_df("JJJJ")


def test_execute_csv_engine_pandas_error() -> None:
    with pytest.raises(ValueError):
        CSVEngine().get_expiry_df("ZSA")


def test_execute_query_from_file_df_caching(db_engine: DatabaseEngine) -> None:
    # Load the DataFrame once
    df1: pl.DataFrame = db_engine.execute_query_from_file("test_query.sql")
    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2: pl.DataFrame = db_engine.execute_query_from_file("test_query.sql")
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time < 0.2
    ), "The DataFrame should be returned right away due to caching"
    assert df1 is df2, "The DataFrame should be cached and return the same object"


def test_execute_query_from_file_df_no_caching(db_engine: DatabaseEngine) -> None:
    # Load the DataFrame once
    df1: pl.DataFrame = db_engine.execute_query_from_file("test_query.sql")
    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2: pl.DataFrame = db_engine.execute_query_from_file_no_cache("test_query.sql")
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time > 0
    ), "The DataFrame should not be returned right away due to no caching"
    assert (
        df1 is not df2
    ), "The DataFrame should not be cached and return the same object"


def test_execute_query_df_caching(db_engine: DatabaseEngine) -> None:
    # Load the DataFrame once
    query = db_engine.load_query("test_query.sql")
    df1: pl.DataFrame = db_engine.execute_query(query)
    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2: pl.DataFrame = db_engine.execute_query(query)
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time < 0.2
    ), "The DataFrame should be returned right away due to caching"
    assert df1 is df2, "The DataFrame should be cached and return the same object"


def test_execute_query_df_no_caching(db_engine: DatabaseEngine) -> None:
    # Load the DataFrame once
    query = db_engine.load_query("test_query.sql")
    df1: pl.DataFrame = db_engine.execute_query(query)
    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2: pl.DataFrame = db_engine.execute_query_no_cache(query)
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time > 0
    ), "The DataFrame not should be returned right away due to no caching"
    assert (
        df1 is not df2
    ), "The DataFrame should not be cached and not return the same object"


def test_load_expiry_returns_correct_dataframe_type(
    csv_engine: CSVEngine,
) -> None:
    """Test that _load_expiry returns the correct DataFrame type based on use_pandas flag."""
    df_pd = csv_engine.get_expiry_df("TestDoc").to_pandas()
    assert isinstance(df_pd, pd.DataFrame), "Expected a Pandas DataFrame"

    df_pl = csv_engine.get_expiry_df("TestDoc")
    assert isinstance(df_pl, pl.DataFrame), "Expected a Polars DataFrame"


def test_load_df_caching(csv_engine: CSVEngine) -> None:
    """Test that _load_df method caches the results."""
    # Load the DataFrame once
    df1 = csv_engine.get_prices_df("TestDoc")

    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2 = csv_engine.get_prices_df("TestDoc")
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time < 0.2
    ), "The DataFrame should be returned right away due to caching"
    assert df1 is df2, "The DataFrame should be cached and return the same object"


def test_load_df_no_caching(csv_engine: CSVEngine) -> None:
    """Test that _load_df method caches the results."""
    # Load the DataFrame once
    df1 = csv_engine.get_prices_df("TestDoc")

    # Modify the internal cache (this requires accessing the cache, which isn't straightforward)
    # Instead, we'll load again and ensure it's the same object due to caching
    start_time = time.time()
    df2 = csv_engine.get_prices_df_no_cache("TestDoc")
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert (
        elapsed_time > 0
    ), "The DataFrame should not be returned right away due to no caching"
    assert (
        df1 is not df2
    ), "The DataFrame should not be cached and not return the same object"


def test_validate_commodity_valid(csv_engine: CSVEngine) -> None:
    """Test that a valid commodity passes validation."""
    try:
        csv_engine.get_prices_df("TestDoc").to_pandas()
    except ValueError:
        pytest.fail("Valid commodity 'TestDoc' raised ValueError unexpectedly!")


def test_validate_commodity_invalid_type(csv_engine: CSVEngine) -> None:
    """Test that passing a non-string commodity raises a TypeError due to typeguard."""
    with pytest.raises(TypeCheckError):
        csv_engine.get_prices_df(
            123
        ).to_pandas()  # Passing integer instead of string


def test_path_setters(csv_engine: CSVEngine) -> None:
    """Test setting new paths using setters."""
    new_expiry_path = "C:/new_expiry_path"
    csv_engine.expiry_path = new_expiry_path
    assert (
        csv_engine.expiry_path == new_expiry_path
    ), "Expiry path was not set correctly"

    new_historical_path = "C:/new_historical_path"
    csv_engine.historical_path = new_historical_path
    assert (
        csv_engine.historical_path == new_historical_path
    ), "Historical path was not set correctly"

    new_current_path = "C:/new_current_path"
    csv_engine.current_path = new_current_path
    assert (
        csv_engine.current_path == new_current_path
    ), "Current path was not set correctly"


def test_set_invalid_path_type(csv_engine: CSVEngine) -> None:
    """Test that setting paths with invalid types raises a TypeError."""
    with pytest.raises(TypeCheckError):
        csv_engine.expiry_path = 12345  # Invalid type

    with pytest.raises(TypeCheckError):
        csv_engine.historical_path = None  # Invalid type

    with pytest.raises(TypeCheckError):
        csv_engine.current_path = ["invalid", "path"]  # Invalid type


def test_clear_cache_db_engine(db_engine: DatabaseEngine) -> None:
    """Test that the cache is cleared in DatabaseEngine."""
    # Load the DataFrame once to populate the cache
    df1 = db_engine.execute_query_from_file("test_query.sql")

    # Clear the cache
    db_engine.clear_all_caches()

    # Load the DataFrame again
    df2 = db_engine.execute_query_from_file("test_query.sql")

    assert df1 is not df2, "The cache should be cleared and return a new object"


def test_clear_cache_csv_engine(csv_engine: CSVEngine) -> None:
    """Test that the cache is cleared in CSVEngine."""
    # Load the DataFrame once to populate the cache
    df1 = csv_engine.get_prices_df("TestDoc")

    # Clear the cache
    csv_engine.clear_all_caches()

    # Load the DataFrame again
    df2 = csv_engine.get_prices_df("TestDoc")
    assert df1 is not df2, "The cache should be cleared and return a new object"


def test_set_empty_current_path(csv_engine: CSVEngine) -> None:
    """Test setting the current path to an empty string and ensure it works."""
    csv_engine.current_path = ""
    try:
        df = csv_engine.get_prices_df("TestDoc").to_pandas()
        assert isinstance(df, pd.DataFrame), "Expected a Pandas DataFrame"
    except Exception as e:
        pytest.fail(f"Setting current path to empty string caused an error: {e}")


def test_set_empty_historical_path(csv_engine: CSVEngine) -> None:
    """Test setting the historical path to an empty string and ensure it works."""
    csv_engine.historical_path = ""
    try:
        df = csv_engine.get_prices_df("TestDoc").to_pandas()
        assert isinstance(df, pd.DataFrame), "Expected a Pandas DataFrame"
    except Exception as e:
        pytest.fail(f"Setting historical path to empty string caused an error: {e}")

def test_date_colummn(csv_engine: CSVEngine) -> None:
    import datetime

    df = csv_engine.get_prices_df("TestDoc")
    date = df.select(pl.col("Date")).max().item()
    assert isinstance(date, datetime.date), "Not a date column"


def test_query_login_timeout() -> None:
    """Test that setting the login timeout to 2 second causes the query to timeout."""
    tester = DatabaseEngine()
    tester.clear_all_caches()
    tester.timeout = 2
    with pytest.raises(sqlalchemy.exc.OperationalError):
        tester.execute_query_from_file("test_query.sql")
