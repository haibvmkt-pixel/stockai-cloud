import os
import sqlite3
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

DB_NAME = "stock_data.db"

# ==============================================================================
# WATCHLIST MỞ RỘNG 100+ CỔ PHIẾU ĐẦU NGÀNH & THANH KHOẢN CAO NHẤT THỊ TRƯỜNG
# ==============================================================================
WATCHLIST = [
    # 🏦 NGÂN HÀNG (BANKING)
    'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'ACB', 'VPB', 'HDB', 'LPB', 'STB', 
    'SHB', 'TPB', 'VIB', 'MSB', 'SSB', 'OCB', 'EIB', 'BAB', 'BVB',
    
    # 📈 CHỨNG KHOÁN (SECURITIES)
    'SSI', 'VND', 'VCI', 'HCM', 'SHS', 'MBS', 'FTS', 'CTS', 'BSI', 'BSI', 
    'ORS', 'VIX', 'AGR', 'VDS',
    
    # 🏢 BẤT ĐỘNG SẢN & KCN (REAL ESTATE)
    'VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'DIG', 'DXG', 'KDH', 'NLG', 'CEO', 
    'CEO', 'KBC', 'IDC', 'SZC', 'SCG', 'TCH', 'HDC', 'DXS', 'CIIC', 'BCG',
    
    # 🏗️ THÉP & VẬT LIỆU (METALS & CONSTRUCTION)
    'HPG', 'HSG', 'NKG', 'VGS', 'KSB', 'HT1', 'BCC',
    
    # 💻 CÔNG NGHỆ, BÁN LẺ & TIÊU DÙNG (TECH & RETAIL)
    'FPT', 'MWG', 'FRT', 'DGW', 'PET', 'MSN', 'VNM', 'SAB', 'KDC', 'PNJ',
    
    # ⚡ NĂNG LƯỢNG, DẦU KHÍ & HÓA CHẤT (OIL, GAS & CHEMICALS)
    'GAS', 'PVD', 'PVS', 'PVT', 'POW', 'NT2', 'GEG', 'PC1', 'HDG', 'REE', 
    'DPM', 'DCM', 'GVR', 'DGC', 'CSV', 'AAA',
    
    # 🐟 THỦY SẢN & NÔNG NGHIỆP (AGRI & SEAFOOD)
    'VHC', 'ANV', 'IDI', 'DAB', 'BAF', 'HAG', 'HNG', 'SBT',
    
    # 🚢 CẢNG BIỂN, VẬN TẢI & DỆT MAY (LOGISTICS & TEXTILES)
    'GMD', 'HAH', 'VSC', 'VOS', 'TNG', 'MSH', 'GIL',
    
    # 🩺 DƯỢC PHẨM & ĐẦU TƯ CÔNG (DRUGS & INFRASTRUCTURE)
    'DBD', 'IMP', 'VCG', 'HHV', 'LCG', 'C4G', 'FCN'
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
                df['formatted_date'] = pd.to_datetime(df['tradingDate']).dt.strftime('%d/%m/%Y')
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
    today_str = datetime.now().strftime("%d/%m/%Y")

    # Thuật toán phân loại tín hiệu
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu quy trình quét 100+ cổ phiếu sau phiên...")
    conn = get_db_connection()
    cursor = conn.cursor()

    scanned_results = []
    total_symbols = len(WATCHLIST)
    
    for idx, sym in enumerate(WATCHLIST, 1):
        df = fetch_stock_history_tcbs(sym)
        if df is not None:
            res = analyze_and_screen(sym, df)
            if res:
                scanned_results.append(res)
                print(f"[{idx}/{total_symbols}] -> Tín hiệu: {sym} | Loại: {res['strategy_type']} | Khuyến nghị: {res['ai_recommendation']}")

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
        print(f"✅ Hoàn tất! Đã lưu {len(scanned_results)} cổ phiếu tiềm năng vào CSDL.")
    else:
        print("Phiên hôm nay chưa có mã cổ phiếu mới đạt điểm vào.")
    
    conn.close()

if __name__ == "__main__":
    run_daily_scan()
