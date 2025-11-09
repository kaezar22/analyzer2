import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json
import re
from utils import summarize_expense, line_chart, bar_chart, pie_chart

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

WORKSHEET_NAME = "cashflow2"

# -------------------- LOAD GOOGLE SHEET --------------------
@st.cache_data(ttl=300)
def load_google_sheets():
    """Load Google Sheet data using credentials stored in Streamlit secrets."""
    try:
        creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        gc = gspread.authorize(creds)

        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        sh = gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(WORKSHEET_NAME)
        values = ws.get_all_values()

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)

        # Convert numeric columns
        if "valor" in df.columns:
            df["valor"] = (
                df["valor"]
                .replace({",": "", "\\$": ""}, regex=True)
                .astype(float)
            )

        return df, None
    except Exception as e:
        return None, str(e)


df, error = load_google_sheets()
if error:
    st.error(f"❌ Error loading Google Sheet: {error}")
    st.stop()
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a financial assistant analyzing a user's spending spreadsheet."}]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a financial assistant analyzing a user's spending spreadsheet."}]
        st.experimental_rerun()

# -------------------- DISPLAY CHAT HISTORY --------------------
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- OPENAI CLIENT --------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------- INTENT DETECTION --------------------
def detect_intent(user_input: str):
    """Detect whether the user wants a summary, plot, or chat."""
    text = user_input.lower()

    # Plot intents
    if any(word in text for word in ["grafica", "gráfico", "plot", "chart", "mostrar tendencia", "visualiza"]):
        if "categoria" in text or "category" in text:
            return "plot_category"
        elif any(w in text for w in ["mes", "tiempo", "trend", "semana", "month"]):
            return "plot_trend"
        elif any(w in text for w in ["pie", "proporción", "porcentaje", "distribución"]):
            return "plot_pie"

    # Summary intent
    if re.search(r"(cu[aá]nto|gasto|gast[ée]|spend|spent|total|ingreso)", text):
        return "summary"

    return "chat"

# -------------------- CHAT INPUT --------------------
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    intent = detect_intent(user_input)

    # -------------------- SUMMARY INTENT --------------------
    if intent == "summary" and df is not None:
        with st.chat_message("assistant"):
            with st.spinner("Calculating..."):
                cat_match = re.search(r"en\s+([a-záéíóúñ]+)", user_input.lower())
                month_match = re.search(
                    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)",
                    user_input.lower()
                )
                category = cat_match.group(1) if cat_match else None
                month = month_match.group(1) if month_match else None

                result = summarize_expense(df, category=category, month=month)
                reply = result["respuesta"]
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # -------------------- CHAT (GPT) INTENT --------------------
    elif intent == "chat":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                df_summary = f"The spreadsheet has {len(df)} rows and the following columns: {', '.join(df.columns)}."
                st.session_state.messages[0]["content"] = f"You are analyzing a financial spreadsheet named '{WORKSHEET_NAME}'. {df_summary}"

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    # -------------------- PLOT INTENTS --------------------
    elif intent in ["plot_trend", "plot_category", "plot_pie"]:
        with st.chat_message("assistant"):
            with st.spinner("Generating chart..."):
                if intent == "plot_trend":
                    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                    df["mes"] = df["fecha"].dt.month_name().str.lower()
                    grouped = df.groupby("mes")["valor"].sum().reset_index()
                    line_chart(grouped, "mes", "valor", "Gasto por mes")

                elif intent == "plot_category":
                    grouped = df.groupby("detalle")["valor"].sum().reset_index().sort_values("valor", ascending=False).head(10)
                    bar_chart(grouped, "detalle", "valor", "Top categorías de gasto")

                elif intent == "plot_pie":
                    grouped = df.groupby("detalle")["valor"].sum().sort_values(ascending=False).head(8)
                    pie_chart(grouped, "Distribución de gastos principales")

                st.success("✅ Chart generated successfully!")
        st.session_state.messages.append({"role": "assistant", "content": "Chart generated successfully!"})
