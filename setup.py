from setuptools import setup, find_packages

setup(
    name= "dbconnect",
    version = "0.16.0",
    include_package_data = True,
    packages= find_packages(),
    author= "Jacob Dwyer",
    author_email="jacobdwyer16@gmail.com",
    description="Database Connection Library for DataFrames from SQL and CSVs",
    python_requires=">=3.10",
    install_requires = [
        "python-dotenv>=1.0.1",
        "pyodbc>=5.1.0",
        "pyarrow>=17.0.0",
        "sqlalchemy>=2.0.34",
        "typeguard>=4.3.0",
    ],
)