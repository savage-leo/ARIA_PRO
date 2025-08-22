import requests, time

API_KEY = "d1312b8fcf3e4d7aa789cb260e825790"
SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY"]

def fetch_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}"
    r = requests.get(url).json()
    return float(r.get("price"))

if __name__ == "__main__":
    print("-------- Quotes --------")
    while True:
        for sym in SYMBOLS:
            try:
                price = fetch_price(sym)
                t = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Quote Symbol: {sym}")
                print(f"Time: {t}")
                print(f"Bid: {price}")
                print(f"Ask: {price+0.0001}")  # simple spread
                print("")
            except Exception as e:
                print("Error:", e)
        time.sleep(5)
