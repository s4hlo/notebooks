
# %%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
dat = yf.Ticker("MSFT")

dat.analyst_price_targets


dat.history(period="1d")


# %%

market = yf.Market('CRYPTOCURRENCIES')

market.summary

market.status


# %%

import yfinance as yf
ticker = yf.Ticker("B3SA3.SA")
history = ticker.history(period="1y")
print(history.head())

# %%

import matplotlib.pyplot as plt

# plot b3 data plot close and open with different colors
plt.figure(figsize=(10, 5))
plt.plot(history['Open'])
plt.plot(history['Close'])
plt.title('B3SA3.SA')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend(['Open', 'Close'])
plt.show()

# %%

# plot in candlestick chart  with seaborn

plt.figure(figsize=(10, 5))
plt.plot(history['Open'])
plt.plot(history['Close'])
plt.title('B3SA3.SA')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend(['Open', 'Close'])
plt.show()

# %%