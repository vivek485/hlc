import pandas as pd
import pytz
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import ta
import time
import asyncio
import aiohttp
import logging
import hashlib
from functools import lru_cache
from contextlib import contextmanager
from asyncio import Semaphore
from aiohttp import TCPConnector
import nest_asyncio
from variable import s 
# assuming this contains your stock symbols
cnb = int(2)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    'interval': 1440,  # enter 15,60,240,1440,10080,43800
    'dayback': 500,
    'timezone': 'Asia/Kolkata',
    'batch_size': 50,
    'semaphore_limit': 10
}

# Initialize global variables
buystock = []
sellstock = []
ist_timezone = pytz.timezone(CONFIG['timezone'])
ed = datetime.now()
stdate = ed - timedelta(days=CONFIG['dayback'])

# Cache implementation
class DataCache:
    def __init__(self):
        self.cache = {}
        self.max_age = timedelta(minutes=15)

    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.max_age:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key, data):
        self.cache[key] = (data, datetime.now())

data_cache = DataCache()

# Utility functions
def conv(x):
    timestamp = int(x.timestamp() * 1000)
    timestamp_str = str(timestamp)[:-4] + '0000'
    return int(timestamp_str)

@lru_cache(maxsize=1000)
def get_cache_key(stock, fromdate, todate, interval):
    return hashlib.md5(f"{stock}{fromdate}{todate}{interval}".encode()).hexdigest()

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logging.info(f'{name} took {elapsed:.2f} seconds')

# Initialize dates and semaphore
fromdate = conv(stdate)
todate = conv(ed)
sem = Semaphore(CONFIG['semaphore_limit'])

async def getdata(session, stock):
    async with sem:
        try:
            cache_key = get_cache_key(stock, fromdate, todate, CONFIG['interval'])
            cached_data = data_cache.get(cache_key)
            if cached_data is not None:
                return cached_data

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            url = f'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/{stock}?endTimeInMillis={todate}&intervalInMinutes={CONFIG["interval"]}&startTimeInMillis={fromdate}'
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logging.error(f"Error fetching data for {stock}: Status {response.status}")
                    return None

                resp = await response.json()
                if not resp.get('candles'):
                    logging.warning(f"No candle data available for {stock}")
                    return None

                # Create DataFrame with optimized dtypes
                dtype_dict = {
                    'Open': 'float32',
                    'High': 'float32',
                    'Low': 'float32',
                    'Close': 'float32',
                    'Volume': 'float32'
                }

                candle = resp['candles']
                dt = pd.DataFrame(candle)
                if dt.empty:
                    logging.warning(f"Empty dataframe for {stock}")
                    return None

                fd = dt.rename(columns={0: 'datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'})
                
                # Validate required columns
                required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in fd.columns for col in required_columns):
                    logging.error(f"Missing required columns for {stock}")
                    return None

                fd['symbol'] = stock
                final_df = fd

                # Convert datatypes efficiently
                for col, dtype in dtype_dict.items():
                    if col in final_df.columns:
                        final_df[col] = final_df[col].astype(dtype)

                # Process datetime
                final_df['datetime1'] = pd.to_datetime(final_df['datetime'], unit='s', utc=True).dt.tz_convert(ist_timezone)
                final_df.set_index('datetime1', inplace=True)
                final_df.drop(columns=['datetime'], inplace=True)

                # Vectorized calculations for shifted values
                shift_cols = {
                    'prevopen': final_df['Open'].shift(1),
                    'prevhigh': final_df['High'].shift(1),
                    'prevlow1': final_df['Low'].shift(2),
                    'prevhigh1': final_df['High'].shift(2),
                    'prevlow2': final_df['Low'].shift(3),
                    'prevhigh2': final_df['High'].shift(3),
                    'prevclose': final_df['Close'].shift(1)
                }
                final_df = final_df.assign(**shift_cols)

                # Technical indicators
                final_df['hlc/3'] = (final_df[['High', 'Low', 'Close']].sum(axis=1)) / 3
                final_df['ema_hlc/3'] = ta.trend.sma_indicator(final_df['hlc/3'], window=5)
                final_df['ema20'] = ta.trend.ema_indicator(final_df['Close'], window=25)
                final_df['atr'] = ta.volatility.average_true_range(final_df['High'], final_df['Low'], final_df['Close'], window=25)
                final_df['kc_lower'] = final_df['ema20'] - (final_df['atr'] * 1)
                final_df['kc_upper'] = final_df['ema20'] + (final_df['atr'] * 1)

                # Generate signals
                final_df['buy_signal'] = np.where(
                    (final_df['Open'] < final_df['ema_hlc/3']) & 
                    (final_df['Close'] > final_df['ema_hlc/3']) &
                    (final_df['Open'] < final_df['kc_lower']) & 
                    (final_df['Close'] > final_df['kc_lower']),
                    1, 0
                )
                
                final_df['sell_signal'] = np.where(
                    (final_df['Open'] > final_df['ema_hlc/3']) & 
                    (final_df['Close'] < final_df['ema_hlc/3']) &
                    (final_df['Open'] > final_df['kc_upper']) & 
                    (final_df['Close'] < final_df['kc_upper']),
                    1, 0
                )

                # Check signals for the last candle
                
                last_candle = final_df.iloc[-cnb]
                if last_candle['buy_signal'] == 1:
                    logging.info(f'Buy signal for {last_candle["symbol"]}')
                    buystock.append(last_candle['symbol'])
                if last_candle['sell_signal'] == 1:
                    logging.info(f'Sell signal for {last_candle["symbol"]}')
                    sellstock.append(last_candle['symbol'])

                data_cache.set(cache_key, final_df)
                return final_df

        except aiohttp.ClientError as e:
            logging.error(f"Network error for {stock}: {str(e)}")
        except ValueError as e:
            logging.error(f"Data processing error for {stock}: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error for {stock}: {str(e)}", exc_info=True)
        return None

async def process_batch(session, stocks_batch):
    tasks = [getdata(session, stock) for stock in stocks_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    with timer("Total execution"):
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=10)
        connector = TCPConnector(limit=50, force_close=True, enable_cleanup_closed=True)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=True
        ) as session:
            results = []
            for i in range(0, len(s), CONFIG['batch_size']):
                batch = s[i:i + CONFIG['batch_size']]
                batch_results = await process_batch(session, batch)
                results.extend([r for r in batch_results if r is not None and not isinstance(r, Exception)])
                await asyncio.sleep(1)  # Rate limiting between batches
            return results

if __name__ == "__main__":
    try:
        nest_asyncio.apply()
        asyncio.run(main())
        st.write(buystocks)
        st.write(sellstock)
        
        print('\nBuy Signals:')
        print(buystock)
        print('\nSell Signals:')
        print(sellstock)
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}", exc_info=True)
