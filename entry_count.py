import pandas as pd, glob

for f in glob.glob("data/*_greedy.csv"):
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    ticker = f.split("/")[-1].replace("_greedy.csv","")
    entry_dates = df[df["entry"] == True][["Close","strategy"]]
    print(f"\n=== {ticker} ({len(entry_dates)}회) ===")
    print(entry_dates.to_string())