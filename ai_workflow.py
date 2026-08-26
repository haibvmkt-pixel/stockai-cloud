import os
import sys
import sqlite3
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import pandas_ta as ta
import schedule
import vnstock

DB_NAME = "stock_data.db"
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1ONMj1-4NmIzoTxSR7BGmGnOYCNcfUXyYiflIcIk2m4U/edit?gid=679189395#gid=679189395"
TARGET_GID = 679189395

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_KEY_FILE = os.path.join(BASE_DIR, "google_credentials.json")

def log_error(msg):
    log_path = os.path.join(os.getcwd(), "error_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def calculate_indicators(df):
    high_129 = df['high'].rolling(window=129).max()
    low_129 = df['low'].rolling(window=129).min()
    df['kijun_129'] = (high_129 + low_129) / 2

    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    tenkan_9 = (high_9 + low_9) / 2

    high_17 = df['high'].rolling(window=17).max()
    low_17 = df['low'].rolling(window=17).min()
    kijun_17 = (high_17 + low_17) / 2

    df['span_a'] = ((tenkan_9 + kijun_17) / 2).shift(17)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['span_b'] = ((high_26 + low_26) / 2).shift(17)

    df['vol_sma20'] = df['volume'].rolling(window=20).mean()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['mfi'] = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=14)

    return df

def analyze_advanced_strategy(df):
    if len(df) < 130:
        return "KHÔNG ĐỦ DỮ LIỆU", 0.50

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = latest['close']
    low_val = latest['low']
    vol = latest['volume']
    vol_avg = latest['vol_sma20'] if pd.notnull(latest['vol_sma20']) else vol
    
    kijun_129 = latest['kijun_129'] if pd.notnull(latest['kijun_129']) else close
    span_a = latest['span_a'] if pd.notnull(latest['span_a']) else close
    span_b = latest['span_b'] if pd.notnull(latest['span_b']) else close
    
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    rsi = latest['rsi'] if pd.notnull(latest['rsi']) else 50.0
    mfi = latest['mfi'] if pd.notnull(latest['mfi']) else 50.0

    prev_close = prev['close']
    prev_cloud_top = max(prev['span_a'], prev['span_b']) if pd.notnull(prev['span_a']) else prev_close

    is_in_cloud = cloud_bottom <= close <= cloud_top
    is_sideway = is_in_cloud or (abs(span_a - span_b) / close < 0.015)

    recent_high = df['high'].tail(60).max()
    is_bounce_from_drop = (recent_high > close * 1.25) and (close > prev_close)

    # 1. TÍN HIỆU BÁN / CẮT LỖ KHẨN CẤP
    entry_candle_low = prev['low']
    if low_val < entry_candle_low or close < (prev_close * 0.95):
        return "BÁN (SELL)", 0.95

    if (prev_close >= prev_cloud_top and close < cloud_top) or (close < cloud_bottom):
        return "BÁN (SELL)", 0.92

    if is_bounce_from_drop:
        return "BÁN (SELL)", 0.88

    if is_sideway and (rsi > 70 or mfi > 75):
        return "BÁN (SELL)", 0.85

    if close > kijun_129 * 1.25 and (vol < vol_avg * 0.6):
        return "BÁN (SELL)", 0.83

    # 2. TÍN HIỆU MUA / VÀO LỆNH
    is_cheap = close <= kijun_129 * 1.02
    is_dry_vol = vol <= vol_avg * 0.55
    is_smart_money = vol >= vol_avg * 2.0

    if is_sideway:
        if is_cheap and (rsi <= 40 or mfi <= 35 or is_dry_vol):
            return "MUA (BUY)", 0.90
        elif is_smart_money and close > prev_close:
            return "MUA (BUY)", 0.85

    if close > cloud_top:
        if is_cheap:
            return "MUA (BUY)", 0.95
        elif is_smart_money and close > prev_close * 1.02:
            return "MUA (BUY)", 0.88
        elif close <= kijun_129 * 1.15:
            return "MUA (BUY)", 0.80

    return "THEO DÕI", 0.55

def push_to_google_sheet_direct(data_list):
    try:
        if not os.path.exists(JSON_KEY_FILE):
            log_error(f"Khong tim thay JSON credentials tai: {JSON_KEY_FILE}")
            return

        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(JSON_KEY_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        
        doc = client.open_by_url(SPREADSHEET_URL)
        try:
            sheet = doc.get_worksheet_by_id(TARGET_GID)
        except Exception:
            sheet = doc.sheet1

        sheet.clear()
        headers = ["NGÀY CẬP NHẬT", "MÃ CP", "GIÁ KHỐP", "RSI (14)", "TÍN HIỆU AI", "ĐỘ TIN CẬY"]
        all_rows = [headers] + data_list
        sheet.update("A1", all_rows)

        total_rows = len(all_rows)
        batch_formats = [
            {
                "range": "A1:F1",
                "format": {
                    "backgroundColor": {"red": 0.08, "green": 0.20, "blue": 0.15},
                    "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "fontSize": 11},
                    "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"
                }
            },
            {
                "range": f"A2:F{total_rows}",
                "format": {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE", "textFormat": {"fontSize": 10}}
            }
        ]

        for idx, row in enumerate(data_list, start=2):
            signal = row[4]
            cell_range = f"E{idx}"
            if "MUA" in signal:
                fmt = {"backgroundColor": {"red": 0.85, "green": 0.95, "blue": 0.85}, "textFormat": {"bold": True, "foregroundColor": {"red": 0.0, "green": 0.5, "blue": 0.0}}}
            elif "BÁN" in signal:
                fmt = {"backgroundColor": {"red": 0.98, "green": 0.85, "blue": 0.85}, "textFormat": {"bold": True, "foregroundColor": {"red": 0.7, "green": 0.0, "blue": 0.0}}}
            else:
                fmt = {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.80}, "textFormat": {"bold": True, "foregroundColor": {"red": 0.6, "green": 0.4, "blue": 0.0}}}
            
            batch_formats.append({"range": cell_range, "format": fmt})

        sheet.batch_format(batch_formats)
    except Exception as e:
        log_error(f"Loi push Google Sheet workflow: {e}")

def get_history_safe(ticker, start, end):
    sources = ['VCI', 'TCBS', 'DNSE']
    for src in sources:
        try:
            if hasattr(vnstock, 'Vnstock'):
                stock = vnstock.Vnstock().stock(symbol=ticker, source=src)
                return stock.quote.history(start=start, end=end, interval='1D')
            elif hasattr(vnstock, 'Quote'):
                quote = vnstock.Quote(symbol=ticker, start_date=start, end_date=end, source=src)
                return quote.history(interval='1D')
        except Exception:
            continue
    return None

def morning_pre_market_analysis():
    try:
        watchlist = ['TCB', 'VCB', 'FPT', 'HPG', 'MBB', 'VNM', 'MSN', 'MWG', 'STB', 'SSI', 'VHM', 'VIC']
        conn = get_db_connection()
        cursor = conn.cursor()
        today_str = datetime.now().strftime("%Y-%m-%d")
        sheet_records = []

        for ticker in watchlist:
            try:
                start_str = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
                df = get_history_safe(ticker, start_str, today_str)
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = calculate_indicators(df)
                    latest_price = float(df.iloc[-1]['close'])
                    latest_rsi = float(df.iloc[-1]['rsi']) if pd.notnull(df.iloc[-1]['rsi']) else 50.0

                    ai_signal, ai_confidence = analyze_advanced_strategy(df)

                    cursor.execute("""
                        INSERT INTO stock_signals (symbol, date, close, rsi, ai_recommendation, ai_confidence, status)
                        VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
                    """, (ticker, today_str, latest_price, latest_rsi, ai_signal, ai_confidence))

                    price_str = f"{latest_price:,.2f}" if latest_price < 1000 else f"{latest_price:,.0f}"
                    sheet_records.append([today_str, ticker, price_str, round(latest_rsi, 1), ai_signal, f"{ai_confidence*100:.0f}%"])
            except Exception as e:
                log_error(f"Loi xu ly ticker {ticker}: {e}")
            
            time.sleep(0.5)

        conn.commit()
        conn.close()

        if sheet_records:
            push_to_google_sheet_direct(sheet_records)
    except Exception as e:
        log_error(f"Loi morning_pre_market_analysis: {e}")

def mid_day_review_and_learn():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today_str = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT id, symbol, close, ai_recommendation FROM stock_signals WHERE date = ? AND status = 'PENDING'", (today_str,))
        records = cursor.fetchall()

        for row in records:
            rec_id, ticker, morning_price, signal = row
            try:
                df_now = get_history_safe(ticker, today_str, today_str)
                if df_now is not None and not df_now.empty:
                    df_now.columns = [c.lower() for c in df_now.columns]
                    mid_price = float(df_now.iloc[-1]['close'])
                    is_accurate = 1 if ("MUA" in signal and mid_price > morning_price) or ("BÁN" in signal and mid_price < morning_price) else 0
                    cursor.execute("UPDATE stock_signals SET status = 'REVIEWED', accuracy_score = ? WHERE id = ?", (is_accurate, rec_id))
            except Exception:
                pass

        conn.commit()
        conn.close()
    except Exception as e:
        log_error(f"Loi mid_day_review: {e}")

schedule.every().day.at("08:30").do(morning_pre_market_analysis)
schedule.every().day.at("11:30").do(mid_day_review_and_learn)

morning_pre_market_analysis()