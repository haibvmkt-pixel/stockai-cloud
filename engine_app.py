import os
import sys
import sqlite3
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests

vnstock_lib = None
try:
    import vnstock3 as vnstock_lib
except Exception:
    try:
        import vnstock as vnstock_lib
    except Exception:
        vnstock_lib = None

DB_NAME = "stock_data.db"

def log_error(msg):
    try:
        log_path = os.path.join(os.getcwd(), "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn

def init_db_and_seed_fast():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    date TEXT,
                    close REAL,
                    rsi REAL,
                    mfi REAL,
                    strategy_type TEXT DEFAULT 'INVESTMENT',
                    ai_recommendation TEXT,
                    ai_confidence REAL,
                    status TEXT DEFAULT 'REVIEWED',
                    accuracy_score INTEGER DEFAULT 1,
                    UNIQUE(symbol, date)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_signals (symbol, date);")
            conn.commit()

            today_vn = datetime.now().strftime("%d/%m/%Y")
            
            inv_stocks = [
                ('MBB', 20.55, 48.5, 52.0, 0.96), ('TCB', 24.50, 42.5, 45.0, 0.95), 
                ('FPT', 132.00, 44.0, 52.0, 0.94), ('HPG', 27.10, 41.2, 48.0, 0.93),
                ('EIB', 17.35, 40.5, 42.0, 0.91), ('MWG', 68.20, 40.5, 42.0, 0.90),
                ('VCB', 91.50, 48.1, 50.0, 0.89), ('MSN', 74.50, 46.0, 51.0, 0.88),
                ('VNM', 66.80, 45.2, 48.0, 0.87), ('ACB', 25.10, 43.0, 46.0, 0.86),
                ('BID', 49.20, 44.5, 47.0, 0.85), ('CTG', 35.40, 42.1, 45.0, 0.84),
                ('HDB', 26.80, 41.0, 43.0, 0.83), ('LPB', 28.50, 45.0, 49.0, 0.82),
                ('VIB', 21.30, 40.2, 42.0, 0.81), ('PNJ', 98.50, 46.0, 50.0, 0.80),
                ('REE', 64.20, 43.5, 45.0, 0.79), ('GAS', 78.50, 41.8, 44.0, 0.78),
                ('VHC', 71.00, 42.0, 46.0, 0.77), ('DGC', 112.0, 44.2, 48.0, 0.76)
            ]
            
            spec_stocks =
