# app.py
import os
import json
import re
from pathlib import Path
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from openai import OpenAI
from utils import line_chart, bar_chart, pie_chart, display_table, aggregate_by_timeframe
from dotenv import load_dotenv
import matplotlib.pyplot as plt


# ---------------------------
# Load environment and set page
# ---------------------------
load_dotenv(override=True)
st.set_page_config(page_title="Cashflow Analyst", layout="wide")


def format_money(value):
    """Format numeric value as currency, e.g. 1234.5 → $1,234.50"""
    try:
        return f"${value:,.2f}"
    except Exception:
        return str(value)


# ---------------------------
# Configuration / Secrets
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

SERVICE_KEY_PATH = os.getenv("key_path")
SERVICE_JSON_STR = os.getenv("ST_SERVICE_ACCOUNT_JSON") or st.secrets.get("ST_SERVICE_ACCOUNT_JSON", None)
SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY") or st.secrets.get("SPREADSHEET_KEY", None)
MEMORY_PATH = Path("memory.json")


# ---------------------------
# Utilities: Google Sheets loader
# ---------------------------
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # 1️⃣ Try JSON from Streamlit secrets (for cloud)
    if SERVICE_JSON_STR:
        try:
            creds_dict = json.loads(SERVICE_JSON_STR)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(creds)
            st.info("✅ Loaded Google credentials from Streamlit secrets JSON.")
            return client
        except Exception as e:
            st.error(f"Error loading credentials from Streamlit secrets JSON: {e}")
            return None

    # 2️⃣ Fallback to local file (for local dev)
    elif SERVICE_KEY_PATH and Path(SERVICE_KEY_PATH).exists():
        try:
            creds = Credentials.from_service_account_file(SERVICE_KEY_PATH, scopes=scopes)
            client = gspread.authorize(creds)
            st.info(f"✅ Loaded Google credentials from file: {SERVICE_KEY_PATH}")
            return client
        except Exception as e:
            st.error(f"Error loading credentials from file: {e}")
            return None

    else:
        st.error("❌ No valid Google credentials found. Check your .env or Streamlit secrets.")
        return None


@st.cache_data(ttl=300)
def load_sheet(sheet_name="cashflow2", spreadsheet_key=None):
    client = get_gspread_client()
    if client is None:
        return None
    if not spreadsheet_key:
        spreadsheet_key = SPREADSHEET_KEY
    if not spreadsheet_key:
        st.error("No spreadsheet key set.")
        return None

    sh = client.open_by_key(spreadsheet_key)
    try:
        worksheet = sh.worksheet(sheet_name)
    except Exception as e:
        st.error(f"Could not open sheet {sheet_name}: {e}")
        return None

    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
        df["mes"] = df["fecha"].dt.strftime("%Y-%m")
        df["mes_name"] = df["fecha"].dt.strftime("%B")
        df["mes_name_es"] = df["fecha"].dt.strftime("%B")
        df["week"] = df["fecha"].dt.isocalendar().week
        df["dow"] = df["fecha"].dt.day_name().str.lower()

    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0)

    for col in ["categoria", "detalle", "medio", "mes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


# ---------------------------
# LLM interpreter
# ---------------------------
def ask_llm_for_intent(question: str):
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = """
    You are an assistant that extracts a compact JSON instruction from a user's question about a cashflow spreadsheet.
    Return ONLY valid JSON with these keys: agg, value_col, filters, timeframe, plot, return_table, explain.

    - agg: "sum" | "avg" | "count"
    - value_col: usually "valor"
    - filters: a dict possibly empty (keys: categoria, detalle, medio, mes, dow, week)
    - timeframe: "this_month", "this_week", "last_month", "month:YYYY-MM", "week:NN"
    - plot: "line" | "bar" | "pie" | null
    - return_table: boolean
    - explain: short sentence
    Columns: fecha, categoria, detalle, valor, medio, mes, dow
    """

    prompt = f"User question: {question}\n\nReturn the JSON only."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        jstart = text.find("{")
        jend = text.rfind("}")
        json_text = text[jstart : jend + 1]
        intent = json.loads(json_text)
        return intent
    except Exception as e:
        st.error(f"LLM error: {e}")
        return None


# ---------------------------
# Execute intent
# ---------------------------
def execute_intent(df: pd.DataFrame, intent: dict):
    if df is None:
        return {"error": "No dataframe loaded"}

    qdf = df.copy()

    filters = intent.get("filters", {}) or {}
    for k, v in filters.items():
        if not v or k not in qdf.columns:
            continue
        val = str(v).strip().lower()
        if k == "mes":
            qdf = qdf[qdf["mes"].astype(str).str.lower() == val]
        elif k in ["detalle", "categoria"]:
            qdf = qdf[qdf[k].astype(str).str.contains(val, na=False)]
        else:
            qdf = qdf[qdf[k].astype(str).str.lower() == val]

    timeframe = intent.get("timeframe")
    now = pd.Timestamp.now()
    if timeframe:
        if timeframe == "this_month":
            qdf = qdf[(qdf["fecha"].dt.month == now.month) & (qdf["fecha"].dt.year == now.year)]
        elif timeframe == "this_week":
            qdf = qdf[(qdf["fecha"].dt.isocalendar().week == now.isocalendar().week)]
        elif timeframe == "last_month":
            last = now - pd.DateOffset(months=1)
            qdf = qdf[(qdf["fecha"].dt.month == last.month)]
        elif timeframe.startswith("month:"):
            mo = timeframe.split("month:")[1]
            qdf = qdf[qdf["mes"] == mo]
        elif timeframe.startswith("week:"):
            wk = int(timeframe.split("week:")[1])
            qdf = qdf[qdf["week"] == wk]

    agg = intent.get("agg", "sum")
    valcol = intent.get("value_col", "valor")
    plot = intent.get("plot")
    return_table = intent.get("return_table", False)
    group_col = intent.get("group_by")

    # --- Grouping ---
    if group_col and group_col in qdf.columns:
        grouped = qdf.groupby(group_col)[valcol]
        if agg == "sum":
            result_series = grouped.sum().sort_values(ascending=False)
        elif agg == "avg":
            result_series = grouped.mean().sort_values(ascending=False)
        else:
            result_series = grouped.count().sort_values(ascending=False)

        if plot == "pie":
            pie_chart(result_series.abs(), f"{agg.title()} of {valcol} by {group_col}")
        elif plot in ["bar", "line"]:
            fig, ax = plt.subplots()
            result_series.plot(kind=plot, ax=ax)
            ax.set_title(f"{agg.title()} of {valcol} by {group_col}")
            st.pyplot(fig)
            plt.close(fig)
        elif return_table:
            st.dataframe(result_series)

        return {"result_value": result_series.sum(), "n_rows": len(qdf), "filtered_df": qdf}

    # --- No grouping ---
    if agg == "sum":
        result_value = qdf[valcol].sum()
    elif agg == "avg":
        result_value = qdf[valcol].mean()
    else:
        result_value = int(qdf.shape[0])

    return {"result_value": result_value, "n_rows": qdf.shape[0], "filtered_df": qdf}


# ---------------------------
# Conversation memory
# ---------------------------
def load_memory():
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(hist):
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Cashflow Analyst — Chat with your Sheet")
st.sidebar.header("Settings & Data")

if st.sidebar.button("Reload sheet"):
    load_sheet.clear()

sheet_tab = "cashflow2"
sheet_key_input = SPREADSHEET_KEY
df = load_sheet(sheet_name=sheet_tab, spreadsheet_key=sheet_key_input)

if df is None:
    st.info("⚠️ Could not load data. Check credentials or sheet permissions.")
else:
    st.sidebar.success(f"✅ Loaded sheet '{sheet_tab}' with {len(df)} rows")

if "history" not in st.session_state:
    st.session_state["history"] = load_memory()

st.subheader("Ask questions about your cashflow")
question = st.text_input("Ask (example: How much did I spend on beer this month?)", key="input_question")
send_clicked = st.button("Send")
response_container = st.container()

if send_clicked:
    with response_container:
        if not question:
            st.warning("Type a question first.")
        else:
            st.session_state["history"].append({"role": "user", "text": question, "time": datetime.now().isoformat()})
            save_memory(st.session_state["history"])

            with st.spinner("Interpreting your question..."):
                intent = ask_llm_for_intent(question)

            if intent is None:
                st.error("Could not interpret question.")
            else:
                outputs = execute_intent(df, intent)
                if "error" in outputs:
                    st.error(outputs["error"])
                else:
                    value = outputs["result_value"]
                    nrows = outputs["n_rows"]
                    filtered = outputs["filtered_df"]

                    text_answer = f"Result: {format_money(value)} ({nrows} rows matched.)"
                    st.success(text_answer)

                    st.session_state["history"].append(
                        {"role": "assistant", "text": text_answer, "time": datetime.now().isoformat()}
                    )
                    save_memory(st.session_state["history"])

                    if intent.get("return_table", False):
                        st.markdown("Filtered rows:")
                        display_table(filtered, n=200)

# Sidebar memory
st.sidebar.header("Conversation history")
for msg in st.session_state.get("history", [])[-20:][::-1]:
    role = msg.get("role", "user")
    prefix = "You: " if role == "user" else "Bot: "
    st.sidebar.write(f"{prefix}{msg.get('text')}")

if st.sidebar.button("Clear memory (session + file)"):
    st.session_state["history"] = []
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink(missing_ok=True)
    st.sidebar.success("Memory cleared.")
