import pandas as pd

# Load CSV files
library_df = pd.read_csv("/Users/elizabethflammer/Desktop/library.csv")   # columns: name, mz
data_df = pd.read_csv("/Users/elizabethflammer/Desktop/data.csv")         # column: mz

mz_tolerance = 0.0002
matches = []

for _, data_row in data_df.iterrows():
    mz_data = data_row["mz"]

    matched_library = library_df[
        (library_df["mz"] >= mz_data - mz_tolerance) &
        (library_df["mz"] <= mz_data + mz_tolerance)
    ]

    for _, lib_row in matched_library.iterrows():
        matches.append({
            "mz_data": mz_data,
            "name": lib_row["name"],
            "mz_library": lib_row["mz"],
            "mz_error": lib_row["mz"] - mz_data
        })

# Create output DataFrame
matched_df = pd.DataFrame(matches)

# Save to CSV
matched_df.to_csv("/Users/elizabethflammer/Desktop/matched_results.csv", index=False)

print("Matching complete. Output saved as matched_results.csv")
