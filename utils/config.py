import streamlit as st

# brand colors
SHEA_BLUE = "#1a5276"
SHEA_GOLD = "#d4a843"
SHEA_DARK = "#0e2f44"
POS_GREEN = "#27ae60"
NEU_YELLOW = "#f1c40f"
NEG_RED = "#c0392b"
NEUTRAL_GRAY = "#85929e"
PALETTE_5 = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1a5276"]

BUILDER_DISPLAY_NAMES = {
    "kb-home": "KB Home",
    "lennar": "Lennar",
    "pulte-homes": "Pulte Homes",
    "shea-homes": "Shea Homes",
}

BUILDER_COLORS = {
    "Shea Homes": "#1a5276",
    "KB Home": "#c0392b",
    "Lennar": "#27ae60",
    "Pulte Homes": "#8e44ad",
}

# pages
PAGES = [
    "Overview",
    "Part 1: Summary Statistics",
    "Part 2: Data Evaluation",
    "Part 3: Sentiment Analysis",
    "Part 4: Advanced NLP",
    "Part 5: Predictive Models",
    "Builder Comparison",
    "Conclusion",
    "Live Prediction Tool",
    "Review Explorer",
]

# global CSS
APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');
    .stApp { background-color: #f7f8fa; }
    .block-container { padding-top: 3rem !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #0e2f44; }
    h1 { font-size: 2.2rem !important; border-bottom: 3px solid #d4a843; padding-bottom: 0.4rem; margin-bottom: 1rem !important; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 16px 20px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    div[data-testid="stMetric"] label { font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; color: #6b7a8d !important; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; font-size: 28px !important; color: #0e2f44 !important; }
    section[data-testid="stSidebar"] { background-color: #0e2f44; }
    section[data-testid="stSidebar"] * { color: #d6eaf8 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(214,234,248,0.15) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #e5e8ec; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans'; font-weight: 500; padding: 10px 22px; color: #6b7a8d; }
    .stTabs [aria-selected="true"] { color: #1a5276 !important; border-bottom: 3px solid #d4a843 !important; }
    .section-header { background: linear-gradient(135deg, #0e2f44 0%, #1a5276 100%); color: white !important; padding: 1.2rem 1.8rem; border-radius: 10px; margin: 2rem 0 1.2rem 0; box-shadow: 0 2px 8px rgba(14,47,68,0.15); }
    .section-header h2 { color: white !important; margin: 0 !important; font-size: 1.5rem !important; border: none !important; padding: 0 !important; }
    .section-header p { color: #d6eaf8 !important; margin: 0.3rem 0 0 0 !important; font-size: 0.95rem; opacity: 0.9; }
    .explain-box { background: white; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0 1.2rem 0; font-size: 0.92rem; line-height: 1.6; color: #2c3e50; }
    .commentary-box { background: #eef4f9; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1.2rem 0 1.8rem 0; font-size: 0.92rem; line-height: 1.7; color: #1c2e3d; }
    .commentary-box b { color: #0e2f44; }
    .static-output { background: #1a1a2e; color: #e0e0e0; padding: 1.2rem 1.5rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; border: 1px solid #2d2d44; margin: 0.5rem 0; }
    .llm-card { background: white; border: 1px solid #e5e8ec; border-radius: 10px; padding: 1.4rem 1.6rem; margin: 1rem 0; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }
    .llm-card h4 { color: #1a5276; margin-top: 0; }
    .finding { background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%); border-left: 4px solid #d4a843; padding: 1rem 1.4rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }
    hr { border: none; border-top: 2px solid #e5e8ec; margin: 2.5rem 0; }
</style>
"""
