import logging
import os
import threading
from functools import lru_cache
from typing import Any, Union

import polars as pl
import pyodbc
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Engine, create_engine, event
from typeguard import typechecked

logger = logging.getLogger(__name__)


class DatabaseEngine:
    """
    DatabaseEngine is a singleton class responsible for managing the database connection and executing queries.
    Attributes:
        _instance (DatabaseEngine): The singleton instance of the class.
        _lock (threading.Lock): A lock to ensure thread-safe singleton initialization.
        _timeout (int): The timeout value for database connections.
        _engine (Union[Engine, None]): The SQLAlchemy engine instance.
        project_root (str): The root directory of the project.
        query_folder (str): The directory where SQL query files are stored.
    Methods:
        __new__(cls, *args, **kwargs): Ensures that only one instance of the class is created.
        _initialize(timeout=300, env_file="db.env"): Initializes the instance with the given timeout and environment file.
        _load_environment(env_file): Loads environment variables from the specified file.
        _initialize_query_folder(): Initializes the query folder from the environment variable.
        timeout: Property to get and set the timeout value.
        engine: Property to get the SQLAlchemy engine instance.
        _create_engine(): Creates and returns the SQLAlchemy engine instance.
        _get_connection_string(): Constructs the database connection string from environment variables.
        load_query(query_filename): Loads an SQL query from a file.
        execute_query(query, **kwargs): Executes the given SQL query and returns the result as a DataFrame.
        execute_query_from_file(query_filename, **kwargs): Executes an SQL query from a file and returns the result as a DataFrame.
        clear_all_caches(): Clears all cached methods in the instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "DatabaseEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, timeout: int = 300, env_file: str = "db.env") -> None:
        self._timeout = timeout
        self._load_environment(env_file)
        self._initialize_query_folder()
        self._engine: Union[Engine, None] = None

    def _load_environment(self, env_file: str) -> None:
        """Load environment variables from the specified file."""
        try:
            env_path = find_dotenv(filename=env_file, raise_error_if_not_found=True)
        except OSError:
            try:
                env_path = find_dotenv(
                    filename=env_file, raise_error_if_not_found=True, usecwd=True
                )
            except OSError as e:
                raise FileNotFoundError(f"Environment file '{env_file}' not found: {e}")
        load_dotenv(env_path)
        self.project_root = os.path.dirname(env_path)

    def _initialize_query_folder(self) -> None:
        self.query_folder = os.getenv("QUERYFOLDER")
        if not self.query_folder:
            raise EnvironmentError(
                "Environment variable 'QUERYFOLDER' is not set in db.env"
            )

        self.query_folder = os.path.join(self.project_root, self.query_folder)

        if not os.path.isdir(self.query_folder):
            raise NotADirectoryError(
                f"The directory specified by QUERYFOLDER does not exist: {self.query_folder}"
            )

    @property
    def timeout(self) -> int:
        return self._timeout

    @timeout.setter
    @typechecked
    def timeout(self, value: int) -> None:
        self._timeout = value
        if hasattr(self, "_engine"):
            self._engine = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        connection_string = self._get_connection_string()
        self._engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            max_overflow=20,
        )

        @event.listens_for(self._engine, "connect")
        def set_query_timeout(
            dbapi_connection: pyodbc.Connection, connection_record: Any
        ) -> None:
            dbapi_connection.timeout = self.timeout

        if not isinstance(self._engine, Engine):
            raise TypeError("self._engine is not type Engine")

        return self._engine

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_connection_string() -> str:
        """Construct the database connection string from environment variables."""
        required_vars = ["DBUSER", "DBPASSWORD", "DBHOST", "DBPORT", "DBNAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing required environment variables from db.env: {','.join(missing_vars)}"
            logging.error(error_msg)
            raise EnvironmentError(error_msg)
        return (
            f"mssql+pyodbc://{os.getenv('DBUSER')}:{os.getenv('DBPASSWORD')}@"
            f"{os.getenv('DBHOST')}:{os.getenv('DBPORT')}/{os.getenv('DBNAME')}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )

    def load_query(self, query_filename: str) -> str:
        if not self.query_folder:
            error_msg = f"The directory specified by QUERYFOLDER does not exist: {self.query_folder}"
            logger.error(error_msg)
            raise NotADirectoryError(error_msg)
        path = os.path.join(self.query_folder, query_filename)
        if os.path.isfile(path):
            try:
                with open(path, "r") as f_:
                    query_str = f_.read()
            except FileNotFoundError as e:
                logger.error(f"Error loading query file {query_filename}: {e}")
        else:
            error_msg = f"SQL query file '{query_filename} not found in directory {self.query_folder}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return query_str

    @lru_cache
    @typechecked
    def execute_query(self, query: str, **kwargs: Any) -> pl.DataFrame:
        df: pl.DataFrame = pl.read_database(query, self.engine, **kwargs)
        return df

    @lru_cache
    @typechecked
    def execute_query_from_file(
        self, query_filename: str, **kwargs: Any
    ) -> pl.DataFrame:
        query = self.load_query(query_filename)
        return self.execute_query(query, **kwargs)

    def clear_all_caches(self) -> None:
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "cache_clear"):
                attr.cache_clear()
        self._initialize()
