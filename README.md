"Coming soon" :)

`CSVEngine`: CSV data
`DatabaseEngine`: database engine

Make sure you have a `db.env` saved **at the root of the directory** with the following parameters:
``` .env
DBUSER=jacob
DBPASSWORD=password
DBHOST=PORTAL
DBPORT=1555
DBNAME=database_name
QUERYFOLDER=query
```

`polars` and `polars-lts-cpu` are mutually exclusive packages that are required for running Polars. `polars` should be the default.

However, the code may generate the following error if the CPU is not within the specification range for `polars`:
```toml

The following required CPU featurese were not detected: avx2, fma, bmi1, bmi2, ...

...

Install the `polars-lts-cpu` package to run polars with better compatibility.
```

This requires the use of a different distribution with Polars, so uninstall `polars` and install `polars-lts-cpu` without any code changes.

To convert `polars` to `pandas`, use `.to_pandas()` at the end of a function.
