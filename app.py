import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from openai import OpenAI
from utils import line_chart, bar_chart, pie_chart, display_table, aggregate_by_timeframe
import matplotlib.pyplot as plt 

st.set_page_config(page_title="Cashflow Analyst", layout="wide")

def format_money(value):
    """
    Format numeric value as currency, e.g. 1234.5 → $1,234.50
    """
    try:
        return f"${value:,.2f}"
    except Exception:
        return str(value)
    
# Configuration / Secrets
# ---------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_CREDENTIALS = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]

WORKSHEET_NAME = "cashflow2"

# --- Load Google Sheets ---
import json
import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

@st.cache_data(ttl=300)
def load_sheet(sheet_name="cashflow2"):
    try:
        # --- Load credentials directly from Streamlit secrets ---
        creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)

        # --- Open the spreadsheet ---
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        sh = client.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)

        # --- Read the sheet data ---
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # --- Clean and enrich the dataframe ---
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
            df['mes'] = df['fecha'].dt.strftime('%Y-%m')
            df['mes_name'] = df['fecha'].dt.strftime('%B')
            df['mes_name_es'] = df['fecha'].dt.strftime('%B')
            df['week'] = df['fecha'].dt.isocalendar().week
            df['dow'] = df['fecha'].dt.day_name().str.lower()

        if 'valor' in df.columns:
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce').fillna(0)

        for col in ['categoria', 'detalle', 'medio', 'mes']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        return df, None

    except Exception as e:
        return None, str(e)
# LLM interpreter
# ---------------------------
def ask_llm_for_intent(question: str):
    # ✅ Use Streamlit secrets instead of os.getenv
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    system = """
    You are an assistant that extracts a compact JSON instruction from a user's question about a cashflow spreadsheet.
    Return ONLY valid JSON with these keys: agg, value_col, filters, timeframe, plot, return_table, explain.

    - agg: "sum" or "avg" or "count"
    - NEVER include the category 'ingreso' of 'categoria' column when the question is about expenses
    - value_col: typically "valor"
    - filters: a dict possibly empty with keys among categoria, detalle, medio, mes, dow, week
    - timeframe: null or a compact time expression (examples: "this_month", "this_week", "last_month", "month:2025-10", "week:2025-42")
    - plot: "line" / "bar" / "pie" / null
    - return_table: boolean - whether the user likely wants a table
    - explain: short explanation in Spanish or English.
      
    Columns: fecha: date data like '10/10/2025' 
             categoria: e.g. 'fun', 'mercado', 'servicios', 'transporte', 'ingreso'
             detalle: e.g. 'beer', 'taxi'
             valor: numeric (negative for expenses, positive for ingresos)
             medio: one of 'bancolombia', 'nequi', 'TC VISA'
             mes: month of the expense (YYYY-MM)
             dow: day of the week
    
    "group_by": null | "categoria" | "medio" | "detalle" | "mes" | "dow" | "week" | null

    Important context:
    - The dataset only includes data from **2025** onward.
    - If the user mentions "this month" or "October", assume they mean "month:2025-10".
    - Keep JSON minimal and valid. Dates always use YYYY-MM or week numbers when appropriate.
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

        # 🧩 Safely extract JSON if there's extra text
        try:
            jstart = text.find('{')
            jend = text.rfind('}')
            json_text = text[jstart:jend + 1]
            intent = json.loads(json_text)
        except Exception:
            intent = json.loads(text)

        return intent

    except Exception as e:
        st.error(f"LLM error: {e}")
        return None
# Data query executor
# ---------------------------
def execute_intent(df: pd.DataFrame, intent: dict):
    """
    Execute aggregation, grouping, and filters based on the LLM intent.
    Returns:
      - result_value
      - n_rows
      - filtered_df
    """
    if df is None or df.empty:
        return {"error": "No dataframe loaded"}

    qdf = df.copy()

    # ---- Apply filters ----
    filters = intent.get("filters", {}) or {}
    for k, v in filters.items():
        if not v:
            continue
        if k in qdf.columns:
            val = str(v).strip().lower()
            if k == "mes":
                qdf = qdf[qdf["mes"].astype(str).str.lower() == val]
            elif k in ["detalle", "categoria"]:
                qdf = qdf[qdf[k].astype(str).str.contains(val, na=False, regex=False)]
            else:
                qdf = qdf[qdf[k].astype(str).str.lower() == val]

    # ---- Timeframe filtering ----
    timeframe = intent.get("timeframe")
    now = pd.Timestamp.now()
    if timeframe and "fecha" in qdf.columns:
        if timeframe == "this_month":
            qdf = qdf[(qdf["fecha"].dt.month == now.month) & (qdf["fecha"].dt.year == now.year)]
        elif timeframe == "this_week":
            qdf = qdf[
                (qdf["fecha"].dt.isocalendar().week == now.isocalendar().week)
                & (qdf["fecha"].dt.year == now.year)
            ]
        elif timeframe == "last_month":
            last = now - pd.DateOffset(months=1)
            qdf = qdf[(qdf["fecha"].dt.month == last.month) & (qdf["fecha"].dt.year == last.year)]
        elif timeframe.startswith("month:"):
            mo = timeframe.split("month:")[1]
            qdf = qdf[qdf["mes"] == mo]
        elif timeframe.startswith("week:"):
            wk = int(timeframe.split("week:")[1])
            qdf = qdf[qdf["week"] == wk]

    agg = intent.get("agg", "sum")
    valcol = intent.get("value_col", "valor")
    plot = intent.get("plot", None)
    return_table = intent.get("return_table", False)

    # ---- Detect group-by column ----
    group_col = intent.get("group_by")
    if not group_col:
        explain = (intent.get("explain") or "").lower()
        if "categor" in explain:
            group_col = "categoria"
        elif "medio" in explain:
            group_col = "medio"
        elif "mes" in explain and "por" in explain:
            group_col = "mes"

    # ---- Aggregate ----
    if group_col and group_col in qdf.columns:
        grouped = qdf.groupby(group_col)[valcol]
        if agg == "sum":
            result_series = grouped.sum().sort_values(ascending=False)
        elif agg == "avg":
            result_series = grouped.mean().sort_values(ascending=False)
        elif agg == "count":
            result_series = grouped.count().sort_values(ascending=False)
        else:
            result_series = grouped.sum().sort_values(ascending=False)

        # ---- Chart or Table ----
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

        return {
            "result_value": result_series.sum(),
            "n_rows": len(qdf),
            "filtered_df": qdf,
        }

    # ---- Default: no grouping, total aggregate ----
    if agg == "sum":
        result_value = qdf[valcol].sum()
    elif agg == "avg":
        result_value = qdf[valcol].mean()
    elif agg == "count":
        result_value = int(qdf.shape[0])
    else:
        result_value = qdf[valcol].sum()

    return {
        "result_value": result_value,
        "n_rows": qdf.shape[0],
        "filtered_df": qdf,
    }
# Streamlit UI
# ---------------------------

st.title("💬 Cashflow Analyst — Chat with your Sheet")
st.sidebar.header("Settings & Data")

# --- Reload button ---
if st.sidebar.button("Reload sheet"):
    load_sheet.clear()  # clear cache

# --- Fixed sheet info ---
sheet_tab = "cashflow2"
sheet_key_input = SPREADSHEET_ID  # from .env

df = load_sheet(sheet_name=sheet_tab, spreadsheet_key=sheet_key_input)

# --- Status messages ---
if df is None:
    st.info("⚠️ Could not load data. Check your service account or sheet permissions.")
else:
    st.sidebar.success(f"✅ Loaded sheet '{sheet_tab}' with {len(df)} rows")


# 🧠 Initialize session memory (now purely in-memory)
if "history" not in st.session_state:
    st.session_state["history"] = []  # No file persistence


# ---------------------------
# Chat UI
# ---------------------------

st.subheader("Ask questions about your cashflow")

# 🟢 Container for the question input
with st.container():
    question = st.text_input("Example: How much did I spend on beer this month?", key="input_question")
    col1, col2 = st.columns([3, 1])
    with col2:
        send_clicked = st.button("Send")

# 🟦 Container for showing results (right below the question block)
response_container = st.container()

if send_clicked:
    with response_container:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            # Add to history (in memory)
            st.session_state["history"].append({
                "role": "user",
                "text": question,
                "time": datetime.now().isoformat()
            })

            # 1️⃣ Interpret via LLM
            with st.spinner("Interpreting your question..."):
                intent = ask_llm_for_intent(question)

            if intent is None:
                st.error("❌ Could not interpret question.")
            else:
                st.markdown("**LLM interpretation:**")
                st.json(intent)

                # 2️⃣ Execute intent
                outputs = execute_intent(df, intent)

                if "error" in outputs:
                    st.error(outputs["error"])
                else:
                    value = outputs["result_value"]
                    nrows = outputs["n_rows"]
                    filtered = outputs["filtered_df"]

                    # 🧩 Format answer
                    agg = intent.get("agg", "sum")
                    if agg == "sum":
                        text_answer = f"Result: {format_money(value)} ({nrows} rows matched.)"
                    elif agg == "avg":
                        text_answer = f"Average: {format_money(value)} ({nrows} rows)."
                    elif agg == "count":
                        text_answer = f"Count: {value} rows."
                    else:
                        text_answer = f"Result: {format_money(value)}"

                    st.success(text_answer)

                    # Save in chat history
                    st.session_state["history"].append({
                        "role": "assistant",
                        "text": text_answer,
                        "time": datetime.now().isoformat()
                    })

                    # Optional: show filtered data
                    if intent.get("return_table", False):
                        st.markdown("### Filtered rows")
                        display_table(filtered, n=200)

                    # Optional: plot visualization
                    plot_type = intent.get("plot")
                    if plot_type == "line" and "fecha" in filtered.columns:
                        daily = filtered.groupby("fecha")[intent.get("value_col", "valor")].sum().reset_index()
                        line_chart(daily, x="fecha", y=intent.get("value_col", "valor"), title="Trend")
                    elif plot_type == "bar" and "categoria" in filtered.columns:
                        agg_cat = filtered.groupby("categoria")[intent.get("value_col", "valor")].sum().reset_index()
                        bar_chart(agg_cat, x="categoria", y=intent.get("value_col", "valor"), title="By Category")
                    elif plot_type == "pie" and "categoria" in filtered.columns:
                        s = filtered.groupby("categoria")[intent.get("value_col", "valor")].sum()
                        pie_chart(s, title="Share by Category")

# ---------------------------
# Show conversation history
# ---------------------------

st.sidebar.header("Conversation history")

# Show last 20 messages (most recent first)
for msg in reversed(st.session_state.get("history", [])[-20:]):
    role = msg.get("role", "user")
    prefix = "🧑 You: " if role == "user" else "🤖 Bot: "
    st.sidebar.write(f"{prefix}{msg.get('text')}")

# Button to clear in-memory history
if st.sidebar.button("🧹 Clear conversation"):
    st.session_state["history"] = []
    st.sidebar.success("Chat memory cleared.")



