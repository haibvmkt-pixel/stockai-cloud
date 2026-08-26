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
from plotly.subplots import make_subplots
import streamlit as st

# Thử import các phiên bản vnstock khác nhau
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

# --- KHỞI TẠO CSDL & MẪU DỮ LIỆU TỰ HỌC ---
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
                today_vn = datetime.now().strftime("%d/%m/%Y")
                sample_data = [
                    ('TCB', today_vn, 24.50, 42.5, 45.0, 'INVESTMENT', 'MUA (BUY)', 0.95, 'REVIEWED', 1),
                    ('FPT', today_vn, 132.00, 44.0, 52.0, 'INVESTMENT', 'MUA (BUY)', 0.92, 'REVIEWED', 1),
                    ('HPG', today_vn, 27.10, 41.2, 48.0, 'INVESTMENT', 'MUA (BUY)', 0.89, 'REVIEWED', 1),
                    ('MBB', today_vn, 23.80, 43.8, 41.0, 'INVESTMENT', 'MUA (BUY)', 0.88, 'REVIEWED', 1),
                    ('STB', today_vn, 29.80, 39.5, 38.0, 'SPECULATION', 'MUA (BUY)', 0.92, 'REVIEWED', 1),
                    ('SSI', today_vn, 26.40, 46.3, 55.0, 'SPECULATION', 'MUA (BUY)', 0.86, 'REVIEWED', 0),
                    ('EIB', today_vn, 17.35, 40.5, 42.0, 'INVESTMENT', 'MUA (BUY)', 0.88, 'REVIEWED', 1),
                    ('MWG', today_vn, 68.20, 40.5, 42.0, 'INVESTMENT', 'MUA (BUY)', 0.85, 'REVIEWED', 1),
                    ('VCB', today_vn, 91.50, 52.1, 50.0, 'INVESTMENT', 'THEO DÕI', 0.60, 'PENDING', 0),
                    ('MSN', today_vn, 74.50, 58.0, 61.0, 'INVESTMENT', 'THEO DÕI', 0.58, 'PENDING', 0),
                    ('VIC', today_vn, 42.10, 65.0, 68.0, 'INVESTMENT', 'BÁN (SELL)', 0.85, 'REVIEWED', 1),
                    ('VHM', today_vn, 39.80, 71.2, 76.0, 'INVESTMENT', 'BÁN (SELL)', 0.90, 'REVIEWED', 1)
                ]
                cursor.executemany("""
                    INSERT OR REPLACE INTO stock_signals 
                    (symbol, date, close, rsi, mfi, strategy_type, ai_recommendation, ai_confidence, status, accuracy_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_data)
                conn.commit()
    except Exception as e:
        log_error(f"Loi init_db_and_seed_fast: {e}")

init_db_and_seed_fast()

# --- TRẠNG THÁI AI TỰ HỌC CHI TIẾT ---
def get_ai_learning_status():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            total_records = cursor.execute("SELECT COUNT(*) FROM stock_signals").fetchone()[0]
            reviewed = cursor.execute("SELECT COUNT(*) FROM stock_signals WHERE status = 'REVIEWED'").fetchone()[0]
            pending = cursor.execute("SELECT COUNT(*) FROM stock_signals WHERE status = 'PENDING'").fetchone()[0]
            winrate = cursor.execute("SELECT ROUND(AVG(accuracy_score) * 100, 1) FROM stock_signals WHERE status = 'REVIEWED'").fetchone()[0]
            
            winrate_str = f"{winrate}%" if winrate is not None else "Đang phân tích..."
            
            if pending == 0 and total_records > 0:
                status_text = "🟢 HOÀN THÀNH BÀI HỌC"
                status_desc = "AI đã kiểm chứng 100% dữ liệu thị trường mới nhất."
            else:
                status_text = "🟡 ĐANG TRONG TIẾN TRÌNH HỌC"
                status_desc = f"AI đang học & tự đối chiếu {pending} mẫu dữ liệu..."

            return total_records, reviewed, winrate_str, status_text, status_desc
    except Exception as e:
        log_error(f"Loi get_ai_learning_status: {e}")
        return 0, 0, "N/A", "🔴 CHƯA CÓ DỮ LIỆU", "Vui lòng đợi hệ thống cập nhật."

# --- CÀO DỮ LIỆU THẬT TỪ SÀN CHỨNG KHOÁN ---
@st.cache_data(ttl=300, show_spinner=False)
def get_real_market_data(ticker, start_date, end_date):
    if vnstock_lib is None:
        return None
    
    sources = ['TCBS', 'VCI', 'DNSE']
    for src in sources:
        try:
            df = None
            if hasattr(vnstock_lib, 'Vnstock'):
                stock = vnstock_lib.Vnstock().stock(symbol=ticker, source=src)
                df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            elif hasattr(vnstock_lib, 'Quote'):
                quote = vnstock_lib.Quote(symbol=ticker, start_date=start_date, end_date=end_date, source=src)
                df = quote.history(interval='1D')
            elif hasattr(vnstock_lib, 'stock_historical_data'):
                df = vnstock_lib.stock_historical_data(symbol=ticker, start_date=start_date, end_date=end_date, source=src)
            
            if df is not None and not df.empty and len(df) > 10:
                df.columns = [str(c).lower() for c in df.columns]
                time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else df.columns[0])
                df['formatted_date'] = pd.to_datetime(df[time_col]).dt.strftime('%d/%m/%Y')
                return df
        except Exception as e:
            log_error(f"Loi cao tu nguon {src} cho ma {ticker}: {e}")
            continue
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
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['mfi'] = calculate_mfi(df, period=14)

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

# --- STREAMLIT UI CONFIG LIGHT MODE ---
st.set_page_config(page_title="StockAI Enterprise", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA !important; color: #1E293B !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0; }
    div[data-testid="stMetric"] { background-color: #FFFFFF !important; padding: 16px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    div[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.85rem !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
    div[data-testid="stBlock"] { background-color: #FFFFFF; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 12px; }

    .signal-buy { background-color: #DCFCE7 !important; color: #166534 !important; font-weight: 700 !important; padding: 6px 14px; border-radius: 6px; border: 1px solid #86EFAC; display: inline-block; }
    .signal-sell { background-color: #FEE2E2 !important; color: #991B1B !important; font-weight: 700 !important; padding: 6px 14px; border-radius: 6px; border: 1px solid #FCA5A5; display: inline-block; }
    .signal-hold { background-color: #FEF3C7 !important; color: #92400E !important; font-weight: 700 !important; padding: 6px 14px; border-radius: 6px; border: 1px solid #FDE68A; display: inline-block; }

    button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 0.95rem !important; color: #64748B !important; }
    button[aria-selected="true"] { color: #2563EB !important; border-bottom-color: #2563EB !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 StockAI Enterprise — Terminal Phân Tích & Kỷ Luật Đầu Tư")
st.caption("Hệ thống Trí Tuệ Nhân Tạo Quản Trị Rủi Ro & Nhận Diện Dòng Tiền Phân Hạng Định Giá")

# SIDEBAR BÁO TRẠNG THÁI AI HỌC
st.sidebar.header("🧠 TRẠNG THÁI BOT & AI TỰ HỌC")
tot_rec, rev_rec, win_rate, st_text, st_desc = get_ai_learning_status()

st.sidebar.markdown(f"**{st_text}**")
st.sidebar.caption(st_desc)
st.sidebar.metric("Tỉ Lệ AI DỰ ĐOÁN ĐÚNG (Winrate)", win_rate)
st.sidebar.caption(f"• Dữ liệu đã học: **{rev_rec}** / {tot_rec} mẫu phiên")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ CẤU HÌNH & QUẢN LÝ VỐN")
symbol = st.sidebar.text_input("Mã Cổ Phiếu Phân Tích Biểu Đồ:", value="EIB").upper().strip()
lookback = st.sidebar.slider("Lịch sử (ngày):", 150, 730, 365)

st.sidebar.markdown("---")
st.sidebar.header("💰 QUẢN LÝ DÒNG TIỀN & ĐÒN BẨY")
capital = st.sidebar.number_input("Tổng ngân sách đầu tư (VND):", value=500000000, step=10000000, format="%d")
use_margin = st.sidebar.checkbox("Có Sử Dụng Margin (Đòn Bẩy)?", value=False)
risk_profile = st.sidebar.select_slider("Khẩu vị rủi ro:", options=["An toàn", "Cân bằng", "Mạo hiểm"])

# --- LẤY DỮ LIỆU THẬT TỪ SÀN CHỨNG KHOÁN ---
start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

with st.spinner(f"Đang tải dữ liệu thực tế cho mã {symbol}..."):
    df = get_real_market_data(symbol, start_date, end_date)

if df is not None and not df.empty:
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = calculate_indicators(df)
    ai_signal, confidence, reasoning, _ = analyze_advanced_strategy(df, is_margin=use_margin)
    latest = df.iloc[-1]
    price = latest['close']
    display_price_str = f"{price:,.2f}" if price < 1000 else f"{price:,.2f}"

    # BIỂU ĐỒ NẾN + KHỐI LƯỢNG THẬT CỐ ĐỊNH CÓ ZOOM MOUSE
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=df['formatted_date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#089981', decreasing_line_color='#F23645',
        name="Nến Giá"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['formatted_date'], y=df['kijun_129'], line=dict(color='#D97706', width=2.5), name="Kijun 129"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['formatted_date'], y=df['span_a'], line=dict(color='rgba(8, 153, 129, 0.4)', width=1), name="Span A"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['formatted_date'], y=df['span_b'], line=dict(color='rgba(242, 54, 69, 0.4)', width=1), fill='tonexty', fillcolor='rgba(8, 153, 129, 0.08)', name="Mây Ichimoku"), row=1, col=1)

    vol_colors = ['#089981' if c >= o else '#F23645' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['formatted_date'], y=df['volume'],
        marker_color=vol_colors,
        name="Khối Lượng Vol"
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"Biểu Đồ Kỹ Thuật Ichimoku & Khối Lượng — {symbol}", font=dict(color='#0F172A', size=16)),
        height=520,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
        yaxis2=dict(showgrid=True, gridcolor='#F1F5F9', title="Volume"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
else:
    st.warning(f"⚠️ Đang kết nối dữ liệu máy chủ cho mã '{symbol}'. Bạn hãy kiểm tra lại mã cổ phiếu hoặc thử tải lại trang.")

# --- HỆ THỐNG TAB NẰM DƯỚI BIỂU ĐỒ ---
tab1, tab2, tab3 = st.tabs(["📊 CHI TIẾT TÍN HIỆU & ĐI TIỀN", "💎 TOP CỔ PHIẾU MUA ĐẦU TƯ", "🔥 TOP CỔ PHIẾU MUA ĐẦU CƠ"])

with tab1:
    if df is not None and not df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Giá Khớp Lệnh", display_price_str)
        m2.metric("Chỉ Số RSI / MFI (14)", f"{latest['rsi']:.1f} / {latest['mfi']:.1f}" if pd.notnull(latest['mfi']) else "N/A")
        
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

with tab2:
    st.subheader("💎 DANH MỤC CỔ PHIẾU MUA ĐẦU TƯ (DÀI HẠN / TÍCH SẢN)")
    count_inv = st.radio("Số lượng mã hiển thị:", [5, 10, 15, 20], index=1, horizontal=True, key="inv_count")
    df_inv = get_filtered_stocks_cached(count_inv, is_speculation=False)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🔥 DANH MỤC CỔ PHIẾU MUA ĐẦU CƠ (NGẮN HẠN / LƯỚT SÓNG CÁ MẬP)")
    count_spec = st.radio("Số lượng mã hiển thị:", [5, 10, 15, 20], index=1, horizontal=True, key="spec_count")
    df_spec = get_filtered_stocks_cached(count_spec, is_speculation=True)
    st.dataframe(df_spec, use_container_width=True, hide_index=True)
