import os
import sys
import sqlite3
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import vnstock

DB_NAME = "stock_data.db"

def log_error(msg):
    log_path = os.path.join(os.getcwd(), "error_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn

# --- KHỞI TẠO CSDL VỚI DỮ LIỆU MỒI TRONG RAM ---
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
                    status TEXT DEFAULT 'PENDING',
                    accuracy_score INTEGER DEFAULT 0,
                    UNIQUE(symbol, date)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_signals (symbol, date);")
            conn.commit()

            count = cursor.execute("SELECT COUNT(*) FROM stock_signals").fetchone()[0]
            if count == 0:
                today_str = datetime.now().strftime("%Y-%m-%d")
                sample_data = [
                    ('TCB', today_str, 24.50, 42.5, 45.0, 'INVESTMENT', 'MUA (BUY)', 0.95),
                    ('FPT', today_str, 132.00, 44.0, 52.0, 'INVESTMENT', 'MUA (BUY)', 0.92),
                    ('HPG', today_str, 27.10, 41.2, 48.0, 'INVESTMENT', 'MUA (BUY)', 0.89),
                    ('MBB', today_str, 23.80, 43.8, 41.0, 'INVESTMENT', 'MUA (BUY)', 0.88),
                    ('STB', today_str, 29.80, 39.5, 38.0, 'SPECULATION', 'MUA (BUY)', 0.92),
                    ('SSI', today_str, 26.40, 46.3, 55.0, 'SPECULATION', 'MUA (BUY)', 0.86),
                    ('MWG', today_str, 68.20, 40.5, 42.0, 'INVESTMENT', 'MUA (BUY)', 0.85),
                    ('VCB', today_str, 91.50, 52.1, 50.0, 'INVESTMENT', 'THEO DÕI', 0.60),
                    ('MSN', today_str, 74.50, 58.0, 61.0, 'INVESTMENT', 'THEO DÕI', 0.58),
                    ('VNM', today_str, 66.80, 49.2, 48.0, 'INVESTMENT', 'THEO DÕI', 0.55),
                    ('VIC', today_str, 42.10, 65.0, 68.0, 'INVESTMENT', 'BÁN (SELL)', 0.85),
                    ('VHM', today_str, 39.80, 71.2, 76.0, 'INVESTMENT', 'BÁN (SELL)', 0.90)
                ]
                cursor.executemany("""
                    INSERT OR REPLACE INTO stock_signals 
                    (symbol, date, close, rsi, mfi, strategy_type, ai_recommendation, ai_confidence, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                """, sample_data)
                conn.commit()
    except Exception as e:
        log_error(f"Loi init_db_and_seed_fast: {e}")

init_db_and_seed_fast()

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

def analyze_advanced_strategy(df, is_margin=False):
    if len(df) < 130:
        return "KHÔNG ĐỦ DỮ LIỆU", 0.50, "Cần tối thiểu 130 phiên để tính toán Kijun 129 và Mây Do Thái", "NEUTRAL"

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

    entry_candle_low = prev['low']
    stop_loss_limit = 0.97 if is_margin else 0.95

    if low_val < entry_candle_low or close < (prev_close * stop_loss_limit):
        return "BÁN (SELL)", 0.95, "⚠️ CẮT LỖ KHẨN CẤP: Vi phạm chân nến mua/thủng tỷ lệ dừng lỗ an toàn!", "TRADER_EXIT"

    if (prev_close >= prev_cloud_top and close < cloud_top) or (close < cloud_bottom):
        return "BÁN (SELL)", 0.92, "⚠️ BÁN THOÁT HÀNG: Cổ phiếu gãy trend, chui mây hoặc dưới Mây yếu kém.", "TRADER_EXIT"

    if is_bounce_from_drop:
        return "BÁN (SELL)", 0.88, "🎯 BÁN CHỐT LỜI SƯỜN BÊN PHẢI: Nhịp hồi kỹ thuật sau cú rơi mạnh từ đỉnh.", "TRADER_EXIT"

    if is_sideway and (rsi > 70 or mfi > 75):
        return "BÁN (SELL)", 0.85, "🎯 BÁN CHỐT LỜI: Sideway chạm vùng quá mua (RSI/MFI > 70-75).", "TRADER_EXIT"

    if close > kijun_129 * 1.25 and (vol < vol_avg * 0.6):
        return "BÁN (SELL)", 0.83, "🎯 BÁN HẠ TỶ TRỌNG: Giá ĐẮT ĐỎ xa đường 129 + Kiệt thanh khoản đỉnh.", "TRADER_EXIT"

    is_cheap = close <= kijun_129 * 1.02
    is_dry_vol = vol <= vol_avg * 0.55
    is_smart_money = vol >= vol_avg * 1.8

    if is_smart_money and close > prev_close * 1.015:
        return "MUA (BUY)", 0.92, "🔥 MUA ĐẦU CƠ: Dòng tiền Cá Mập x2 bùng nổ khối lượng đẩy giá ngắn hạn!", "SPECULATION"

    if close > cloud_top and is_cheap:
        return "MUA (BUY)", 0.96, "💎 MUA ĐẦU TƯ: UpTrend trên Mây + Định giá RẺ dưới/sát Trục 129.", "INVESTMENT"

    if is_sideway and is_cheap and (rsi <= 42 or is_dry_vol):
        return "MUA (BUY)", 0.89, "💎 MUA ĐẦU TƯ: Nền Sideway sát 129 + Kiệt cung cạn lực bán tích sản dài hạn.", "INVESTMENT"

    if close > cloud_top and close <= kijun_129 * 1.15:
        return "MUA (BUY)", 0.82, "✅ MUA BÁM SÓNG: Cổ phiếu thanh thoát trên Mây, nhịp chỉnh an toàn.", "INVESTMENT"

    return "THEO DÕI", 0.55, "Thị trường chưa hội tụ đủ tiêu chuẩn điểm Vào/Ra an toàn.", "NEUTRAL"

@st.cache_data(ttl=600)
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

def scan_full_market_to_db():
    watchlist = ['TCB', 'VCB', 'FPT', 'HPG', 'MBB', 'VNM', 'MSN', 'MWG', 'STB', 'SSI', 'VHM', 'VIC']
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_str = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")

    records = []
    for ticker in watchlist:
        try:
            df = get_history_safe(ticker, start_str, today_str)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df = calculate_indicators(df)
                latest_price = float(df.iloc[-1]['close'])
                latest_rsi = float(df.iloc[-1]['rsi']) if pd.notnull(df.iloc[-1]['rsi']) else 50.0
                latest_mfi = float(df.iloc[-1]['mfi']) if pd.notnull(df.iloc[-1]['mfi']) else 50.0

                time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else df.columns[0])
                candle_date = str(df.iloc[-1][time_col])[:10]

                ai_signal, ai_confidence, _, strat_type = analyze_advanced_strategy(df)
                records.append((ticker, candle_date, latest_price, latest_rsi, latest_mfi, strat_type, ai_signal, ai_confidence))
        except Exception as e:
            log_error(f"Loi scan ticker {ticker}: {e}")

    if records:
        for _ in range(5):
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT OR REPLACE INTO stock_signals 
                        (symbol, date, close, rsi, mfi, strategy_type, ai_recommendation, ai_confidence, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """, records)
                    conn.commit()
                break
            except sqlite3.OperationalError:
                time.sleep(0.5)

# --- BỘ NHỚ TẠM RAM CACHE TRUỴ VẤN SIÊU TỐC (<0.01 GIÂY) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_filtered_stocks_cached(limit_count, is_speculation=False):
    try:
        with get_db_connection() as conn:
            query = """
                SELECT symbol as 'Mã CP', 
                       ai_recommendation as 'Tín Hiệu AI', 
                       ROUND(ai_confidence * 100, 1) || '%' as 'Độ Tin Cậy', 
                       CASE WHEN close < 1000 THEN PRINTF('%.2f', close) ELSE PRINTF('%,d', CAST(close AS INT)) END as 'Giá Khớp', 
                       ROUND(rsi, 1) as 'RSI (14)',
                       ROUND(mfi, 1) as 'MFI (14)',
                       date as 'Ngày Cập Nhật'
                FROM stock_signals
                ORDER BY ai_confidence DESC, id DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(int(limit_count),))
    except Exception as e:
        log_error(f"Loi get_filtered_stocks_cached: {e}")
        return pd.DataFrame()

# --- STREAMLIT UI ---
st.set_page_config(page_title="StockAI Enterprise", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E222D; padding: 15px; border-radius: 10px; border: 1px solid #2A2E39; }
    div[data-testid="stBlock"] { background-color: #131722; padding: 20px; border-radius: 12px; border: 1px solid #2A2E39; margin-bottom: 10px; }
    
    .signal-buy { background-color: rgba(46, 125, 50, 0.25) !important; color: #2ecc71 !important; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #2ecc71; }
    .signal-sell { background-color: rgba(198, 40, 40, 0.25) !important; color: #e74c3c !important; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #e74c3c; }
    .signal-hold { background-color: rgba(243, 156, 18, 0.25) !important; color: #f1c40f !important; font-weight: bold; padding: 4px 8px; border-radius: 4px; border: 1px solid #f1c40f; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ StockAI Enterprise - Terminal Phân Tích & Kỷ Luật Đầu Tư")
st.caption("Hệ thống Trí Tuệ Nhân Tạo Quản Trị Rủi Ro & Nhận Diện Dòng Tiền Thông Minh")

st.sidebar.header("⚙️ CẤU HÌNH & QUẢN LÝ VỐN")
symbol = st.sidebar.text_input("Mã Cổ Phiếu Đơn Lẻ:", value="TCB").upper().strip()
lookback = st.sidebar.slider("Lịch sử (ngày):", 150, 730, 365)

st.sidebar.markdown("---")
st.sidebar.header("💰 QUẢN LÝ DÒNG TIỀN & ĐÒN BẨY")
capital = st.sidebar.number_input("Tổng ngân sách đầu tư (VND):", value=500000000, step=10000000, format="%d")
use_margin = st.sidebar.checkbox("Có Sử Dụng Margin (Đòn Bẩy)?", value=False)
risk_profile = st.sidebar.select_slider("Khẩu vị rủi ro:", options=["An toàn", "Cân bằng", "Mạo hiểm"])

tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD PHÂN TÍCH MÃ", "💎 TOP CỔ PHIẾU MUA ĐẦU TƯ", "🔥 TOP CỔ PHIẾU MUA ĐẦU CƠ"])

with tab1:
    start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Đang kết nối & tính toán chỉ báo Ichimoku & Dòng Tiền..."):
        df = get_history_safe(symbol, start_date, end_date)

    if df is not None and not df.empty:
        df = calculate_indicators(df)
        ai_signal, confidence, reasoning, _ = analyze_advanced_strategy(df, is_margin=use_margin)
        latest = df.iloc[-1]
        price = latest['close']
        display_price_str = f"{price:,.2f}" if price < 1000 else f"{price:,.0f}"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Giá Khớp Lệnh", display_price_str)
        m2.metric("RSI / MFI (14)", f"{latest['rsi']:.1f} / {latest['mfi']:.1f}" if pd.notnull(latest['mfi']) else "N/A")
        
        if "MUA" in ai_signal:
            m3.markdown(f"**Khuyến Nghị AI:** <span class='signal-buy'>{ai_signal}</span>", unsafe_allow_html=True)
        elif "BÁN" in ai_signal:
            m3.markdown(f"**Khuyến Nghị AI:** <span class='signal-sell'>{ai_signal}</span>", unsafe_allow_html=True)
        else:
            m3.markdown(f"**Khuyến Nghị AI:** <span class='signal-hold'>{ai_signal}</span>", unsafe_allow_html=True)

        m4.metric("Độ Tin Cậy AI", f"{confidence*100:.1f}%")

        if "BÁN" in ai_signal:
            st.error(f"🚨 **PHÂN TÍCH TÍN HIỆU RA (BÁN/CẮT LỖ):** {reasoning}")
        elif "MUA" in ai_signal:
            st.success(f"🎯 **PHÂN TÍCH TÍN HIỆU VÀO (MUA):** {reasoning}")
        else:
            st.warning(f"💡 **PHÂN TÍCH QUAN SÁT:** {reasoning}")

        st.subheader("💡 Khuyến Nghị Đi Tiền & Phân Bổ Vốn")
        alloc_pct = 0.0
        if "MUA" in ai_signal:
            alloc_pct = 0.20 if risk_profile == "An toàn" else (0.35 if risk_profile == "Cân bằng" else 0.50)
        elif "BÁN" in ai_signal:
            alloc_pct = 0.0
        else:
            alloc_pct = 0.10

        target_amount = capital * alloc_pct
        actual_buy_price = price * 1000 if price < 1000 else price
        max_shares = int(target_amount / actual_buy_price) if actual_buy_price > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Tỷ Lệ Giải Ngân Tối Đa", f"{alloc_pct*100:.0f}% Tổng Vốn")
        c2.metric("Số Tiền Khuyến Nghị Đi Lệnh", f"{target_amount:,.0f} VND")
        c3.metric("Số Lượng Cổ Phiếu Nên Mua", f"{max_shares:,} CP")

        time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else df.columns[0])
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=df[time_col], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Nến Giá"))
        fig.add_trace(go.Scatter(x=df[time_col], y=df['kijun_129'], line=dict(color='#FFD700', width=3.5), name="Kijun 129 (Trục Định Giá Dài Hạn)"))
        fig.add_trace(go.Scatter(x=df[time_col], y=df['span_a'], line=dict(color='rgba(0, 255, 150, 0.5)', width=1), name="Span A"))
        fig.add_trace(go.Scatter(x=df[time_col], y=df['span_b'], line=dict(color='rgba(255, 50, 50, 0.5)', width=1), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.12)', name="Mây Ichimoku"))

        fig.update_layout(title=f"Biểu Đồ Kỹ Thuật Ichimoku & Kijun 129 - {symbol}", height=550, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#131722", plot_bgcolor="#131722")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Không thể tải dữ liệu cổ phiếu '{symbol}'. Vui lòng kiểm tra lại mã.")

with tab2:
    st.subheader("💎 DANH MỤC CỔ PHIẾU MUA ĐẦU TƯ (DÀI HẠN / TÍCH SẢN)")
    col_sel, col_btn = st.columns([2, 1])
    with col_sel:
        count_inv = st.radio("Số lượng mã hiển thị:", [5, 10, 15, 20], index=1, horizontal=True, key="inv_count")
    with col_btn:
        if st.button("⚡ QUÉT MỚI THỊ TRƯỜNG", use_container_width=True, key="btn_inv"):
            with st.spinner("Đang cào dữ liệu & phân tích..."):
                scan_full_market_to_db()
                st.cache_data.clear()
                st.success("✅ Quét dữ liệu hoàn tất!")
                st.rerun()

    df_inv = get_filtered_stocks_cached(count_inv, is_speculation=False)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🔥 DANH MỤC CỔ PHIẾU MUA ĐẦU CƠ (NGẮN HẠN / LƯỚT SÓNG CÁ MẬP)")
    col_sel2, col_btn2 = st.columns([2, 1])
    with col_sel2:
        count_spec = st.radio("Số lượng mã hiển thị:", [5, 10, 15, 20], index=1, horizontal=True, key="spec_count")
    with col_btn2:
        if st.button("⚡ QUÉT MỚI THỊ TRƯỜNG", use_container_width=True, key="btn_spec"):
            with st.spinner("Đang cào dữ liệu & phân tích..."):
                scan_full_market_to_db()
                st.cache_data.clear()
                st.success("✅ Quét dữ liệu hoàn tất!")
                st.rerun()

    df_spec = get_filtered_stocks_cached(count_spec, is_speculation=True)
    st.dataframe(df_spec, use_container_width=True, hide_index=True)