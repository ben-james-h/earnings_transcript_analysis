import os
import json
import pandas as pd
import yfinance as yf
from openai import OpenAI
from prompt_template import earnings_extraction_prompt
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Map company names to tickers
TICKER_MAP = {
    "BlackRock": "BLK",
    "T. Rowe Price Group": "TROW",
    "Franklin Templeton": "BEN",
    "Invesco": "IVZ",
    "AllianceBernstein": "AB",
    "State Street Global Advisors": "STT",
    "Janus Henderson": "JHG",
    "Affiliated Managers Group": "AMG",
    "Federated Hermes": "FHI",
    "WisdomTree": "WT"}

#Load in earnings transcript
def load_transcript(filepath):
    with open(filepath, "r") as f:
        return f.read()

#See prompt template - retrieves key info in JSON
def extract_data(transcript_text,filename):

    prompt = earnings_extraction_prompt(transcript_text,filename)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)

    except Exception as e:
        print("Invalid JSON returned by model")
        print(raw_output)
        print(e)
        return None

#Use yfinance to compute stock returns + abnormal returns, for CAR. We will split into getting price data,...
#...stock returns and abnormal returns separately.

def get_price_data(ticker, earnings_date, window=10):

    try:
        event_date = pd.to_datetime(earnings_date)

        start = event_date - pd.Timedelta(days=window)
        end = event_date + pd.Timedelta(days=window)

        stock = yf.download(ticker, start=start, end=end, progress=False)
        market = yf.download("^GSPC", start=start, end=end, progress=False)

        if stock.empty or market.empty:
            raise ValueError("No market data returned")

        # Handle both possible column structures
        if "Adj Close" in stock.columns:
            stock_prices = stock["Adj Close"]
        else:
            stock_prices = stock["Close"]

        if "Adj Close" in market.columns:
            market_prices = market["Adj Close"]
        else:
            market_prices = market["Close"]

        if isinstance(stock_prices, pd.DataFrame):
            stock_prices = stock_prices.squeeze()

        if isinstance(market_prices, pd.DataFrame):
            market_prices = market_prices.squeeze()

        # Align dates safely
        df = pd.concat(
            [stock_prices.rename("stock"), market_prices.rename("market")],
            axis=1
        ).dropna()

        # Compute returns
        df["stock_return"] = df["stock"].pct_change()
        df["market_return"] = df["market"].pct_change()

        df = df.dropna()

        if len(df) < 6:
            raise ValueError("Not enough trading data")

        return df

    except Exception as e:
        print(f"\nError retrieving price data for {ticker}")
        print(f"Earnings date: {earnings_date}")
        print(e)
        return None


def compute_return(df, event_date,window=1):
    try:
        event_loc = df.index.get_indexer([event_date],method = 'nearest')[0]
        if event_loc - window < 0 or event_loc + window >= len(df):
            return None

        start_price = df.iloc[event_loc-window]["stock"]
        end_price = df.iloc[event_loc+window]["stock"]

        return (end_price / start_price) - 1

    except Exception as e:
        print(f"Error computing stock return")
        print(e)
        return None

def compute_event_study(df, event_date, window=1):
    try:
        event_loc = df.index.get_indexer([event_date],method='nearest')[0]

        abnormal = df["stock_return"] - df["market_return"]
        window_data = abnormal.iloc[event_loc-window:event_loc+window+1]

        return window_data.sum()

    except Exception as e:
        print("\nError computing event study")
        print(e)
        return None



#Full function - passes functions through each transcript and saves JSON/total CSV output
if __name__ == "__main__":
    all_results = []

    for file in sorted(os.listdir("data/raw")):
                           
        if not file.endswith(".txt"):
            continue
        
        filepath = os.path.join("data/raw", file)
        filename = os.path.basename(filepath)

        parts = filename.replace(".txt","").split("_")

        quarter = None
        year = None
        #To ensure data parsed about by quarter and year
        for p in parts:
            if p.lower().startswith("q"):
                quarter = p.upper()
            if p.isdigit() and len(p) == 4:
                year = int(p)

        print(f"\nProcessing transcript: {file}")

        transcript = load_transcript(filepath)
        result = extract_data(transcript,filename)

        if result is None:
            print("Extraction failed")
            continue

        result["quarter"] = quarter
        result["year"] = year
        result["transcript_source"] = filename

        # Retrieve company name
        company = result.get("company")
        
        if company is None:
            filename_lower = filename.lower()

            for name in TICKER_MAP.keys():
                if name.lower().replace(" ", "") in filename_lower.replace("_", ""):
                    company = name
                    print(f"Company inferred from filename: {company}")
                    result["company"] = company
                    break

        # Map company to ticker
        ticker = TICKER_MAP.get(company)
        earnings_date = result.get("earnings_date")

        if earnings_date is None:
            print("Attempting fallback earnings date")

            if year and quarter:
                quarter_month_map = {
                    "Q1": "04-15",
                    "Q2": "07-15",
                    "Q3": "10-15",
                    "Q4": "01-15"
                }

                earnings_date = f"{year}-{quarter_month_map.get(quarter,'01-15')}"
                result['earnings_date'] = earnings_date

        if ticker and earnings_date:
            price_data = get_price_data(ticker,earnings_date)
            if price_data is None:
                print(f"Skipping return calculations for {ticker}")
                stock_return = None
                car_3 = None
                car_5 = None
            else:

                event_date = pd.to_datetime(earnings_date)
                stock_return = compute_return(price_data,event_date,window=1)

                car_3 = compute_event_study(price_data, event_date, window=1)
                car_5 = compute_event_study(price_data, event_date, window=2)

            result["ticker"] = ticker
            result["earnings_reaction"] = stock_return
            result["CAR_3d"] = car_3
            result["CAR_5d"] = car_5

        else:
            print(f"No ticker or earnings date found for {company}")

        print("\nEXTRACTED DATA\n")
        print(json.dumps(result, indent=4))

        os.makedirs("outputs", exist_ok=True)

        company_name = result.get("company", "unknown_company")
        if company_name is None:
            company_name_clean = "unknown_company"
        else:
            company_name_clean = company_name.lower().replace(" ", "_").replace(".", "")

        json_filename = f"outputs/{company_name_clean}_{quarter}_{year}.json"

        with open(json_filename, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\nSaved output to {json_filename}")

        all_results.append(result)
    
    if len(all_results) > 0:

        os.makedirs("data/processed", exist_ok=True)

        df = pd.DataFrame(all_results)
        df = df.apply(pd.to_numeric, errors="ignore")

        csv_filename = "data/processed/asset_manager_earnings_dataset.csv"

        df.to_csv(csv_filename, index=False)

        print(f"\nSaved combined dataset to {csv_filename}")
    else:
        print("No results were generated.")