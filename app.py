# app.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json
import re

# your plotting / helper functions
from utils import summarize_expense, line_chart, bar_chart, pie_chart, display_table, aggregate_by_timeframe

# -------------------- Page setup --------------------
st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

WORKSHEET_NAME = "cashflow2"

# -------------------- Load Google Sheet --------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets():
    try:
        # Load credentials directly from Streamlit secrets (exactly like your old working version)
        creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)

        # Authorize gspread
        gc = gspread.authorize(creds)

        # Spreadsheet ID from secrets
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        sh = gc.open_by_key(spreadsheet_id)

        # Load worksheet
        ws = sh.worksheet(WORKSHEET_NAME)
        values = ws.get_all_values()

        if not values:
            return None, "Worksheet is empty."

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)

        # ---- Safe column cleaning (the only change from your original safe loader) ----
        # strip whitespace from string columns
        df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)

        # Convert fecha to datetime if present
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

        # Safely convert valor to numeric:
        # - remove common currency chars (commas, $)
        # - coerce errors to NaN
        # - fill NaN with 0 so sums work
        if "valor" in df.columns:
            df["valor"] = (
                df["valor"]
                .astype(str)
                .str.replace(r"[^\d\-\.\,]", "", regex=True)  # remove non-numeric except comma/dot/minus
                .str.replace(",", "", regex=True)  # remove thousands comma if present
            )
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


# Load sheet
df, error = load_google_sheets()
if error:
    st.error(f"Error loading Google Sheet: {error}")
    st.stop()
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")
    st.dataframe(df)  # Show the DataFrame above chat


# -------------------- Initialize chat memory --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]
        st.experimental_rerun()


# -------------------- Initialize OpenAI --------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -------------------- Display previous messages --------------------
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------- Intent detection (simple) --------------------
def detect_intent(user_input: str):
    text = user_input.lower()
    # plotting keywords
    if any(word in text for word in ["grafica", "gráfico", "plot", "chart", "mostrar tendencia", "visualiza", "tendencia"]):
        if "categoria" in text or "detalle" in text or "category" in text:
            return "plot_category"
        elif any(w in text for w in ["mes", "tiempo", "trend", "month", "semana", "mes pasado"]):
            return "plot_trend"
        elif any(w in text for w in ["pie", "porción", "proporción", "porcentaje", "distribución", "composición"]):
            return "plot_pie"

    # summary / numeric question
    if re.search(r"(cu[aá]nto|gasto|gast[ée]|gastaste|ingres|total|spend|spent)", text):
        return "summary"

    return "chat"


# -------------------- Chat input --------------------
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    # Add user message to memory and UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    intent = detect_intent(user_input)

    # If it's a numeric summary request, call local summarizer
    if intent == "summary":
        with st.chat_message("assistant"):
            with st.spinner("Calculating..."):
                # try to detect a category and a month (spanish)
                cat_match = re.search(r"en\s+([a-záéíóúñ]+)", user_input.lower())
                month_match = re.search(
                    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)",
                    user_input.lower(),
                )
                category = cat_match.group(1) if cat_match else None
                month = month_match.group(1) if month_match else None

                # summarize_expense expects df with 'fecha','detalle','valor', etc.
                try:
                    result = summarize_expense(df, category=category, month=month)
                    reply = result.get("respuesta") if isinstance(result, dict) else str(result)
                except Exception as e:
                    reply = f"Error computing summary: {e}"

                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # If it's a plotting request, use utils plotting functions
    elif intent in ("plot_trend", "plot_category", "plot_pie"):
        with st.chat_message("assistant"):
            with st.spinner("Generating chart..."):
                try:
                    if intent == "plot_trend":
                        # ensure fecha is datetime
                        if "fecha" not in df.columns:
                            st.error("No 'fecha' column available to plot trend.")
                        else:
                            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                            df["mes"] = df["fecha"].dt.strftime("%Y-%m")
                            trend = df.groupby("mes")["valor"].sum().reset_index()
                            line_chart(trend, "mes", "valor", title="Tendencia de gastos por mes")

                    elif intent == "plot_category":
                        if "detalle" in df.columns:
                            grouped = df.groupby("detalle")["valor"].sum().reset_index().sort_values("valor", ascending=False).head(20)
                            bar_chart(grouped, "detalle", "valor", title="Gastos por detalle (top 20)")
                        else:
                            st.error("No 'detalle' column available to plot categories.")

                    elif intent == "plot_pie":
                        if "detalle" in df.columns:
                            s = df.groupby("detalle")["valor"].sum().sort_values(ascending=False).head(8)
                            pie_chart(s, title="Composición de gastos (top 8)")
                        else:
                            st.error("No 'detalle' column available for pie chart.")

                    reply = "Chart generated."
                    st.success("✅ Chart generated successfully!")
                except Exception as e:
                    reply = f"Chart error: {e}"
                    st.error(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Otherwise, fall back to full LLM chat
    else:
        # Add spreadsheet summary context to system message so LLM knows about columns
        if df is not None:
            df_summary = f"The spreadsheet has {len(df)} rows and the following columns: {', '.join(df.columns)}."
            st.session_state.messages[0]["content"] = (
                f"You are analyzing a financial spreadsheet named '{WORKSHEET_NAME}'. {df_summary}"
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                except Exception as e:
                    reply = f"LLM error: {e}"
                    st.error(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
