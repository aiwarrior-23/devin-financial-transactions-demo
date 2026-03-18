"""
Streamlit-based Financial Transaction Analysis Chatbot.

A professional chat interface for analyzing financial transactions,
detecting fraud patterns, and exploring transaction data.
"""

import os
import sys

# Ensure the repo root is on sys.path so that the 'chat' package is importable
# regardless of the working directory Streamlit uses.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import pandas as pd

from chat.analyzer import load_data, get_enriched_data, get_summary_statistics
from chat.chatbot import process_query


# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Transaction Analyzer",
    page_icon="\U0001f4b0",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_custom_css(theme: str) -> str:
    """Return custom CSS based on the selected theme."""
    if theme == "Dark":
        return """
        <style>
            :root {
                --bg-primary: #0e1117;
                --bg-secondary: #1a1d23;
                --bg-card: #1e2228;
                --text-primary: #e6e9ef;
                --text-secondary: #a3a8b4;
                --accent: #4f8bf9;
                --accent-light: #6c9dfa;
                --border: #2d3139;
                --success: #2ecc71;
                --warning: #f39c12;
                --danger: #e74c3c;
                --user-msg-bg: #2b3a55;
                --bot-msg-bg: #1e2228;
                --input-bg: #1a1d23;
                --shadow: rgba(0, 0, 0, 0.3);
            }

            .stApp {
                background-color: var(--bg-primary);
            }

            section[data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border);
            }

            .main-header {
                background: linear-gradient(135deg, #1a2332 0%, #2b3a55 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                margin-bottom: 1.5rem;
                border: 1px solid var(--border);
                box-shadow: 0 4px 12px var(--shadow);
            }

            .main-header h1 {
                color: var(--accent-light);
                font-size: 1.8rem;
                margin: 0;
                font-weight: 700;
            }

            .main-header p {
                color: var(--text-secondary);
                margin: 0.3rem 0 0 0;
                font-size: 0.95rem;
            }

            .chat-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 1rem;
                border-radius: 12px;
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                margin-bottom: 1rem;
            }

            .user-message {
                background: var(--user-msg-bg);
                border: 1px solid #3a4d6e;
                border-radius: 12px 12px 4px 12px;
                padding: 0.8rem 1.2rem;
                margin: 0.5rem 0;
                color: var(--text-primary);
            }

            .bot-message {
                background: var(--bot-msg-bg);
                border: 1px solid var(--border);
                border-radius: 12px 12px 12px 4px;
                padding: 0.8rem 1.2rem;
                margin: 0.5rem 0;
                color: var(--text-primary);
            }

            .stat-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 2px 8px var(--shadow);
            }

            .stat-card h3 {
                color: var(--accent-light);
                font-size: 1.6rem;
                margin: 0;
            }

            .stat-card p {
                color: var(--text-secondary);
                font-size: 0.85rem;
                margin: 0.3rem 0 0 0;
            }

            .risk-high { color: #e74c3c; font-weight: 700; }
            .risk-medium { color: #f39c12; font-weight: 700; }
            .risk-low { color: #2ecc71; font-weight: 700; }

            .sidebar-info {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }

            div[data-testid="stChatMessage"] {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.5rem;
            }

            .stDataFrame {
                border: 1px solid var(--border);
                border-radius: 8px;
            }

            div[data-testid="stExpander"] {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
            }
        </style>
        """
    else:
        return """
        <style>
            :root {
                --bg-primary: #f8f9fc;
                --bg-secondary: #ffffff;
                --bg-card: #ffffff;
                --text-primary: #1a1d23;
                --text-secondary: #6b7280;
                --accent: #2563eb;
                --accent-light: #3b82f6;
                --border: #e5e7eb;
                --success: #059669;
                --warning: #d97706;
                --danger: #dc2626;
                --user-msg-bg: #eff6ff;
                --bot-msg-bg: #f9fafb;
                --input-bg: #ffffff;
                --shadow: rgba(0, 0, 0, 0.06);
            }

            .stApp {
                background-color: var(--bg-primary);
            }

            section[data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border);
            }

            .main-header {
                background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                margin-bottom: 1.5rem;
                border: 1px solid var(--border);
                box-shadow: 0 4px 12px var(--shadow);
            }

            .main-header h1 {
                color: var(--accent);
                font-size: 1.8rem;
                margin: 0;
                font-weight: 700;
            }

            .main-header p {
                color: var(--text-secondary);
                margin: 0.3rem 0 0 0;
                font-size: 0.95rem;
            }

            .chat-container {
                max-height: 600px;
                overflow-y: auto;
                padding: 1rem;
                border-radius: 12px;
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                margin-bottom: 1rem;
            }

            .user-message {
                background: var(--user-msg-bg);
                border: 1px solid #bfdbfe;
                border-radius: 12px 12px 4px 12px;
                padding: 0.8rem 1.2rem;
                margin: 0.5rem 0;
                color: var(--text-primary);
            }

            .bot-message {
                background: var(--bot-msg-bg);
                border: 1px solid var(--border);
                border-radius: 12px 12px 12px 4px;
                padding: 0.8rem 1.2rem;
                margin: 0.5rem 0;
                color: var(--text-primary);
            }

            .stat-card {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 2px 8px var(--shadow);
            }

            .stat-card h3 {
                color: var(--accent);
                font-size: 1.6rem;
                margin: 0;
            }

            .stat-card p {
                color: var(--text-secondary);
                font-size: 0.85rem;
                margin: 0.3rem 0 0 0;
            }

            .risk-high { color: #dc2626; font-weight: 700; }
            .risk-medium { color: #d97706; font-weight: 700; }
            .risk-low { color: #059669; font-weight: 700; }

            .sidebar-info {
                background: #f1f5f9;
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }

            div[data-testid="stChatMessage"] {
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.5rem;
            }

            .stDataFrame {
                border: 1px solid var(--border);
                border-radius: 8px;
            }

            div[data-testid="stExpander"] {
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: 10px;
            }
        </style>
        """


def render_sidebar(enriched: pd.DataFrame, stats: dict) -> str:
    """Render the sidebar with dataset info and controls."""
    with st.sidebar:
        st.markdown("### Settings")
        theme = st.selectbox(
            "Theme",
            ["Dark", "Light"],
            index=0,
            help="Switch between dark and light mode",
        )

        st.markdown("---")

        st.markdown("### Dataset Overview")

        st.markdown(
            """
            <div class="sidebar-info">
                <strong>Transactions:</strong> {}<br>
                <strong>Total Volume:</strong> ${:,.2f}<br>
                <strong>Avg Amount:</strong> ${:,.2f}
            </div>
            """.format(
                stats["total_transactions"],
                stats["total_amount"],
                stats["avg_amount"],
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### Risk Distribution")
        risk_dist = stats.get("risk_distribution", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            high_count = risk_dist.get("HIGH", 0)
            st.metric("HIGH", high_count)
        with col2:
            medium_count = risk_dist.get("MEDIUM", 0)
            st.metric("MEDIUM", medium_count)
        with col3:
            low_count = risk_dist.get("LOW", 0)
            st.metric("LOW", low_count)

        st.markdown("---")

        st.markdown("### Quick Actions")

        if st.button("Show High-Risk", use_container_width=True, type="primary"):
            st.session_state.quick_query = "Show high-risk transactions"

        if st.button("Show Fraud Flagged", use_container_width=True):
            st.session_state.quick_query = "Show fraud-flagged transactions"

        if st.button("Show Summary", use_container_width=True):
            st.session_state.quick_query = "Show summary statistics"

        if st.button("Show Anomalies", use_container_width=True):
            st.session_state.quick_query = "Show balance anomalies"

        if st.button("Top 10 Riskiest", use_container_width=True):
            st.session_state.quick_query = "Show top 10 riskiest transactions"

        st.markdown("---")

        st.markdown("### Example Questions")
        st.markdown(
            """
- Show high-risk transactions
- Explain transaction #3
- Show CASH_OUT transactions
- Transactions above 100000
- Show balance anomalies
- Show summary statistics
            """
        )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return theme


def render_header():
    """Render the main page header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>\U0001f4b0 Financial Transaction Analyzer</h1>
            <p>AI-powered chatbot for analyzing financial transactions, detecting fraud patterns, and exploring risk signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats_cards(stats: dict):
    """Render summary statistic cards at the top."""
    risk_dist = stats.get("risk_distribution", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            """
            <div class="stat-card">
                <h3>{}</h3>
                <p>Total Transactions</p>
            </div>
            """.format(stats["total_transactions"]),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="stat-card">
                <h3>${:,.0f}</h3>
                <p>Total Volume</p>
            </div>
            """.format(stats["total_amount"]),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="stat-card">
                <h3><span class="risk-high">{}</span></h3>
                <p>High Risk</p>
            </div>
            """.format(risk_dist.get("HIGH", 0)),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="stat-card">
                <h3><span class="risk-medium">{}</span></h3>
                <p>Medium Risk</p>
            </div>
            """.format(risk_dist.get("MEDIUM", 0)),
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div class="stat-card">
                <h3><span class="risk-low">{}</span></h3>
                <p>Low Risk</p>
            </div>
            """.format(risk_dist.get("LOW", 0)),
            unsafe_allow_html=True,
        )

    st.markdown("")


def style_risk_level(val: str) -> str:
    """Apply color styling to risk level values."""
    colors = {
        "HIGH": "color: #e74c3c; font-weight: bold",
        "MEDIUM": "color: #f39c12; font-weight: bold",
        "LOW": "color: #2ecc71; font-weight: bold",
    }
    return colors.get(val, "")


def display_dataframe(df: pd.DataFrame):
    """Display a styled dataframe."""
    display_cols = [
        "transaction_id",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "risk_score",
        "risk_level",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()

    display_df["amount"] = display_df["amount"].apply(lambda x: "${:,.2f}".format(x))

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "transaction_id": st.column_config.NumberColumn("ID", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "amount": st.column_config.TextColumn("Amount", width="medium"),
            "nameOrig": st.column_config.TextColumn("Origin", width="medium"),
            "nameDest": st.column_config.TextColumn("Destination", width="medium"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
                width="medium",
            ),
            "risk_level": st.column_config.TextColumn("Risk Level", width="small"),
        },
    )


def main():
    """Main application entry point."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "enriched_data" not in st.session_state:
        df = load_data()
        st.session_state.enriched_data = get_enriched_data(df)
        st.session_state.stats = get_summary_statistics(st.session_state.enriched_data)

    enriched = st.session_state.enriched_data
    stats = st.session_state.stats

    # Render sidebar and get theme
    theme = render_sidebar(enriched, stats)

    # Apply theme CSS
    st.markdown(get_custom_css(theme), unsafe_allow_html=True)

    # Main content
    render_header()
    render_stats_cards(stats)

    st.markdown("---")

    # Chat interface
    st.markdown("### Chat with the Analyzer")

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if "dataframe" in message and message["dataframe"] is not None:
                with st.expander("View Data Table", expanded=True):
                    display_dataframe(message["dataframe"])

    # Handle quick action queries from sidebar
    if "quick_query" in st.session_state and st.session_state.quick_query:
        query = st.session_state.quick_query
        st.session_state.quick_query = None

        st.session_state.messages.append({"role": "user", "content": query})

        response_text, response_df = process_query(query, enriched)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "dataframe": response_df,
            }
        )
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about financial transactions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response_text, response_df = process_query(prompt, enriched)

        with st.chat_message("assistant"):
            st.markdown(response_text)
            if response_df is not None:
                with st.expander("View Data Table", expanded=True):
                    display_dataframe(response_df)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "dataframe": response_df,
            }
        )


if __name__ == "__main__":
    main()
