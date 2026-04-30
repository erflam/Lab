import pandas as pd
from pathlib import Path

#THIS IS FOR REMOVING ROWS THAT CONTAINING SIGNIFICANT ZEROS AFTER BLK FILTERING

# ---- User settings ----
INPUT  = Path('MZmine.xlsx')
OUTPUT = Path('MZmine ZF.xlsx')
THRESHOLD = 0.75  # drop rows with >= 75% zeros (ignoring the first column)
EXCEL_SHEET = 0   # sheet index or name when reading Excel
# -----------------------

def load_table(path: Path, sheet=0) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == '.csv':
        # If you need a specific delimiter, add sep=',' here
        return pd.read_csv(path)
    elif suf in ('.xlsx', '.xls'):
        # Requires openpyxl (for .xlsx) or xlrd (older .xls)
        return pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported input format: {suf}")

def save_table(df: pd.DataFrame, path: Path):
    suf = path.suffix.lower()
    if suf == '.csv':
        df.to_csv(path, index=False)
    elif suf in ('.xlsx', '.xls'):
        # For .xlsx this uses openpyxl by default
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suf}")

def main():
    # Load table (CSV or Excel)
    df = load_table(INPUT, sheet=EXCEL_SHEET)

    # Determine how many metadata columns to ignore (up to 2 if available)
    meta_cols = min(1, df.shape[1])

    # Separate metadata and data
    meta = df.iloc[:, :meta_cols]
    data = df.iloc[:, meta_cols:]

    # Convert data to numeric; non-numeric -> NaN
    num = data.apply(pd.to_numeric, errors='coerce')

    # Count zeros and numeric cells per row (in the data portion only)
    zero_count = (num == 0).sum(axis=1)
    numeric_count = num.notna().sum(axis=1)

    # Proportion of zeros across numeric cells
    # If a row has no numeric cells, treat prop_zero as 0.0 (i.e., keep)
    prop_zero = zero_count.div(numeric_count).fillna(0.0)

    # Keep rows with < THRESHOLD zeros
    mask = prop_zero < THRESHOLD
    cleaned = df.loc[mask]

    print(f"Dropping {(~mask).sum()} of {len(df)} rows "
          f"(threshold: {THRESHOLD:%} zeros; ignoring first {meta_cols} column(s)).")

    # Save table (CSV or Excel, matching OUTPUT extension)
    save_table(cleaned, OUTPUT)

if __name__ == "__main__":
    main()
