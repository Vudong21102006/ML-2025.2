import pandas as pd, io, re

def read_wine_csv(path):
    """
    The wine CSV files have a quirky format where the entire header line
    is wrapped in outer quotes and inner column names use "" as escape.
    This function reads them correctly.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    
    # First line: strip the outer quotes, then replace "" with nothing
    header_raw = lines[0].strip()
    # Remove leading/trailing quote if present
    if header_raw.startswith('"') and header_raw.endswith('"'):
        header_raw = header_raw[1:-1]
    # Replace doubled quotes "" with empty, then split on ;
    cols = [c.replace('""','').strip('"') for c in header_raw.split(';')]
    
    # Data lines are straightforward sep=;
    data_text = "".join(lines[1:])
    df = pd.read_csv(io.StringIO(data_text), sep=";", header=None, names=cols)
    return df

# Test
df_red = read_wine_csv(r"data\raw\winequality-red.csv")
df_white = read_wine_csv(r"data\raw\winequality-white.csv")
print("Red  :", df_red.shape, df_red.columns.tolist())
print("White:", df_white.shape, df_white.columns.tolist())
print(df_red.head(2))
