"""
Smart Energy Anomaly Detector — Streamlit Application.

Main entry point for the Streamlit frontend. Implements a multi-page
navigation with a custom dark theme sidebar.

Run with:
    streamlit run app/main.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Smart Energy Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark theme ──────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border-right: 1px solid rgba(148,163,184,0.08);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #E2E8F0;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E2E8F0 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #E2E8F0;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30,41,59,0.6);
        border-radius: 8px;
        color: #E2E8F0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
    }

    /* Selectbox / Input styling */
    .stSelectbox [data-baseweb="select"],
    .stDateInput input,
    .stNumberInput input {
        background-color: #1E293B;
        color: #E2E8F0;
        border-color: rgba(148,163,184,0.15);
    }

    /* DataTable */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(148,163,184,0.1);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    /* Animated glow for sidebar title */
    .sidebar-title {
        background: linear-gradient(135deg, #6366F1, #06B6D4, #10B981);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h1 class="sidebar-title">⚡ Energy AI</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color: #64748B; font-size: 0.85rem; margin-bottom: 2rem;">'
        'Smart Anomaly Detection System</p>',
        unsafe_allow_html=True,
    )

    # Navigation
    page = st.radio(
        "Navigation",
        options=["📊 Dashboard", "🧠 Smart Alerts", "🔬 Exploration"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color: #475569; font-size: 0.75rem; line-height: 1.8;">
            <b>About</b><br>
            Powered by Isolation Forest<br>
            & Prophet forecasting<br><br>
            <b>Data</b><br>
            UCI Household Power<br>
            Consumption Dataset<br><br>
            <b>Status</b><br>
            <span style="color: #10B981;">● Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page routing ───────────────────────────────────────────────────
if page == "📊 Dashboard":
    from app.pages.dashboard import render
    render()
elif page == "🧠 Smart Alerts":
    from app.pages.smart_alerts import render
    render()
elif page == "🔬 Exploration":
    from app.pages.exploration import render
    render()
