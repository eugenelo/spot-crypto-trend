# Converting Tick to OHLCV Data

## Converting tick data CSV files

Tick data can be converted into OHLCV data of arbitrary time frame using the binary `//data:tick_to_ohlc`. Resampling will always be done with respect to the UTC timezone.

### Single Input Path

To convert a single tick data CSV file into a single 1d OHLCV CSV file, use
```
bazel run //data:tick_to_ohlc -- --input_path {input_path} --output_path {output_path} --timeframe 1d
```

### Input Directory

This binary also accepts as input file directories containing multiple tick data CSV files. Behavior is highly configurable, with options to (1) recurse through subdirectories, (2) combine output from multiple input files into a single output file, (3) skip overwriting existing output files, and more.

Some potential use cases are listed below.

#### Convert all tick data from a single folder into a single OHLCV CSV

```
bazel run //data:tick_to_ohlc -- --input_dir {input_dir} --output_path {output_path} --timeframe 1d --combine
```

#### Convert tick data in a nested file structure into one OHLCV CSV per ticker

This assumes that your tick data is organized as described in [Fetching Kraken Tick Data - Output Format](./fetching-kraken-tick-data.md#output-format).

```
bazel run //data:tick_to_ohlc -- --input_dir {input_dir} --output_dir {output_dir} --timeframe 1d --recursive --combine
```

#### Convert tick data in a nested file structure into one OHLCV CSV per tick CSV

Unlike the combined recursive case above, this makes no assumptions on your directory structure. Each CSV file is operated on independently.
```
bazel run //data:tick_to_ohlc -- --input_dir {input_dir} --output_dir {output_dir} --timeframe 1d --recursive
```

However, this does require that your tick data files is properly timezone-aligned. For example, if you wanted to produce daily OHLCV data in UTC and your tick data was stored in monthly files in GMT-4 (4 hours behind UTC), the binary would produce two different OHLCV entries for the last day of each month, one from 20 hours of data and the other from 4 hours of data, both of which are incorrect. This would be difficult to reconcile without returning to the raw tick data.


### Automatic Updates

An existing dataset of partially processed tick data can be incrementally updated with minimal redundant processing. This assumes that your tick data is organized as described in [Fetching Kraken Tick Data - Output Format](./fetching-kraken-tick-data.md#output-format).

```
bazel run //data:tick_to_ohlc -- --input_dir {input_dir} --output_dir {output_dir} --timeframe 1d --auto
```

- Specifying `--auto` implicitly specifies `--recursive` and `--overwrite`, excluding `--combine`.
  - The input file directory is recursively traversed to find all tick data CSV files.
- Each tick CSV is only (re-)processed if it potentially contains new data that would impact the OHLCV outputs.
  - To avoid loading and checking each file, we use a heuristic of processing all files which potentially contain data within 7 days of the current date.
  - The start and end days of each file's data window is determined from the filename, **not** the file contents.


## Combining OHLCV CSV files

Multiple OHLCV CSV files can also be combined into a single CSV via the binary `//data:combine_ohlc`.

### Single output path

```
bazel run //data:combine_ohlc -- --input_dir {input_dir} --data_frequency 1d --output_path {output_path}
```

- The `data_frequency` / `timeframe` of all input OHLCV CSVs must match the input `--data_frequency` arg
- Recursive iteration through `{input_dir}` is possible with the `--recursive` flag

### Dropping incomplete rows

Incomplete rows can be detected by checking the latest available timestamp with the bin cutoff. For example, a last trade time of `2024-06-10T02:30:00+0` and a `data_frequency` of `1h` would indicate that the `2024-06-10T02:00:00+0` OHLCV row is potentially incomplete. These rows can be dropped with the `--drop_incomplete_rows` (analogous to [`--drop_last_row` from fetching](./fetching-kraken-ohlcv-data.md#dropping-incomplete-rows)).

Note that this behavior uses the current UTC time for determining whether a row is incomplete. Sticking with our example of last trade time `2024-06-10T02:30:00+0` and `data_frequency=1h`, if the current time were `2024-06-10T02:59:00+0` then this row would be dropped. 2 minutes later (current time `2024-06-10T03:01:00+0`) and the row would be kept. If using this feature, it's imperative to **update your tick data immediately prior**, and preferably after the bin cutoff (e.g. with `1h` bins, update tick data after the turn of the hour).


### Automatic Updates

Signal / position generation currently expect OHLCV data in a single CSV file, formatted as a long dataframe with many vertically stacked `(<datetime>, <ticker>)` rows. Completely recreating this from individual `(ticker, month)` OHLCV files on every tick data update would be unnecessarily slow, as only the latest data would require updating.

In an ideal world, both our tick data and our generated OHLCV data would be stored in separate tables in a relational database. Ingesting newly fetched tick data would involve simply inserting new rows into the trades table and updating / inserting the exact OHLCV rows for which updates were necessary. With our CSV file setup, we can approximate this solution by consolidating the individual `(ticker, month)` OHLCV files into one combined OHLCV file per month. Automatic updates would involve recursively iterating over all `(ticker, month)` data files, as before, but (re-)combining only those files for the latest month based on the current date.


```
bazel run //data:combine_ohlc -- --input_dir {input_dir} --data_frequency 1d --output_dir {output_dir} --auto
```

This produces combined files of the form
```
...
combined_2024-04-01_2024-05-01_OHLC.csv
combined_2024-05-01_2024-06-01_OHLC.csv
combined_2024-06-01_2024-07-01_OHLC.csv
```

where `combined_2024-06-01_2024-07-01_OHLC.csv` contains all OHLCV rows for all tickers with `datetime` between `2024-06-01T00:00:00+0` inclusive and `2024-07-01T00:00:00+0` exclusive.


Producing the final combined file would require an additional [single output path combine](#single-output-path) of all of these monthly consolidated files.
