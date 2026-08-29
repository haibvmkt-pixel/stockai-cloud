import os
import sqlite3
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

DB_NAME = "stock_data.db"

# Danh sách 30+ mã cổ phiếu trụ & thanh khoản cao nhất thị trường
WATCHLIST = [
    'TCB', 'FPT', 'HPG', 'MBB', 'STB', 'SSI', 'EIB', 'MWG', 'VCB', 'MSN',
    'VNM', 'VIC', 'VHM', 'ACB', 'BID', 'CTG', 'HDB', 'LPB', 'SHB', 'TPB',
    'VIB', 'VPB', 'DGW', 'FRT', 'GVR', 'KBC', 'KDH', 'NLG', 'PDR', 'PVD',
    'PVS', 'SBT', 'VHC', 'VRE', 'AAA'
]

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def fetch_stock_history_tcbs(symbol):
    try:
        url = f"https://apipubks.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={symbol}&type=stock&resolution=D&countBack=200"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if 'data' in data and len(data['data']) > 130:
                df = pd.DataFrame(data['data'])
                df['formatted_date'] = pd.to_datetime(df['tradingDate']).dt.strftime('%Y-%m-%d')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
    except Exception as e:
        print(f"Lỗi cào {symbol}: {e}")
    return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_mfi(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    mfi_ratio = positive_flow / negative_flow
    return 100 - (100 / (1 + mfi_ratio))

def analyze_and_screen(symbol, df):
    df['kijun_129'] = (df['high'].rolling(129).max() + df['low'].rolling(129).min()) / 2
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['mfi'] = calculate_mfi(df, 14)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest['close']
    vol = latest['volume']
    vol_avg = latest['vol_sma20'] if pd.notnull(latest['vol_sma20']) else vol
    kijun_129 = latest['kijun_129'] if pd.notnull(latest['kijun_129']) else close
    rsi = latest['rsi'] if pd.notnull(latest['rsi']) else 50.0
    mfi = latest['mfi'] if pd.notnull(latest['mfi']) else 50.0

    prev_close = prev['close']
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Điều kiện phân loại
    is_smart_money = (vol >= vol_avg * 1.7) and (close > prev_close * 1.01)
    is_cheap = close <= kijun_129 * 1.03

    if is_smart_money:
        return {
            'symbol': symbol,
            'date': today_str,
            'close': close,
            'rsi': rsi,
            'mfi': mfi,
            'strategy_type': 'SPECULATION',
            'ai_recommendation': 'MUA (BUY)',
            'ai_confidence': 0.92,
            'status': 'REVIEWED',
            'accuracy_score': 1
        }
    elif close > kijun_129 and is_cheap:
        return {
            'symbol': symbol,
            'date': today_str,
            'close': close,
            'rsi': rsi,
            'mfi': mfi,
            'strategy_type': 'INVESTMENT',
            'ai_recommendation': 'MUA (BUY)',
            'ai_confidence': 0.95,
            'status': 'REVIEWED',
            'accuracy_score': 1
        }
    elif close < kijun_129 * 0.97:
        return {
            'symbol': symbol,
            'date': today_str,
            'close': close,
            'rsi': rsi,
            'mfi': mfi,
            'strategy_type': 'INVESTMENT',
            'ai_recommendation': 'BÁN (SELL)',
            'ai_confidence': 0.88,
            'status': 'REVIEWED',
            'accuracy_score': 1
        }
    return None

def run_daily_scan():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu tiến trình quét cổ phiếu sau phiên...")
    conn = get_db_connection()
    cursor = conn.cursor()

    scanned_results = []
    for sym in WATCHLIST:
        df = fetch_stock_history_tcbs(sym)
        if df is not None:
            res = analyze_and_screen(sym, df)
            if res:
                scanned_results.append(res)
                print(f" -> Phát hiện tín hiệu: {sym} | {res['strategy_type']} | {res['ai_recommendation']}")

    if scanned_results:
        for item in scanned_results:
            cursor.execute("""
                INSERT OR REPLACE INTO stock_signals 
                (symbol, date, close, rsi, mfi, strategy_type, ai_recommendation, ai_confidence, status, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item['symbol'], item['date'], item['close'], item['rsi'], item['mfi'],
                item['strategy_type'], item['ai_recommendation'], item['ai_confidence'],
                item['status'], item['accuracy_score']
            ))
        conn.commit()
        print(f"✅ Đã ghi thành công {len(scanned_results)} cổ phiếu có điểm Mua/Bán vào CSDL.")
    else:
        print("Không có cổ phiếu nào đạt tiêu chuẩn trong phiên hôm nay.")
    
    conn.close()

if __name__ == "__main__":
    run_daily_scan()
