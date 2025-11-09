# app.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json
import re

# Import your helper functions
from utils import summarize_expense, line_chart, bar_chart, pie_chart, display_table

# --- Streamlit page setup ---
st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

WORKSHEET_NAME = "cashflow2"

# --- Load Google Sheets ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets():
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

        # Try to cast numeric + datetime columns
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)


# --- Load sheet ---
df, error = load_google_sheets()
if error:
    st.error(f"Error loading Google Sheet: {error}")
    st.stop()
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")
    st.dataframe(df)


# --- Initialize chat memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]


# --- Initialize OpenAI ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# --- Display chat history ---
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Helper: detect plot or summary requests ---
def detect_intent(user_input: str):
    """Very basic pattern matcher for intent classification."""
    text = user_input.lower()
    if any(word in text for word in ["grafica", "gráfico", "plot", "chart", "mostrar tendencia"]):
        if "categoria" in text:
            return "plot_category"
        elif "mes" in text or "tiempo" in text or "trend" in text:
            return "plot_trend"
        elif "pie" in text or "proporción" in text:
            return "plot_pie"
    elif any(word in text for word in ["gasto", "gasté", "cuánto gasté", "ingreso", "total"]):
        return "summary"
    return "chat"


# --- Chat input ---
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Detect intent
    intent = detect_intent(user_input)

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # --- SUMMARY REQUEST ---
            if intent == "summary":
                # Try to extract category/month if mentioned
                cat_match = re.search(r"en (\w+)", user_input.lower())
                month_match = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", user_input.lower())

                category = cat_match.group(1) if cat_match else None
                month = month_match.group(1) if month_match else None

                result = summarize_expense(df, category=category, month=month)
                reply = result["respuesta"]
                st.markdown(reply)

            # --- PLOT REQUESTS ---
            elif intent == "plot_category":
                st.markdown("📊 Here's a breakdown of expenses by category:")
                grouped = df.groupby("categoria")["valor"].sum().reset_index()
                bar_chart(grouped, "categoria", "valor", "Gastos por categoría")
                reply = "Aquí tienes la gráfica de gastos por categoría."

            elif intent == "plot_trend":
                st.markdown("📈 Here's a trend of your expenses over time:")
                trend = df.groupby("mes")["valor"].sum().reset_index()
                line_chart(trend, "mes", "valor", "Tendencia de gastos por mes")
                reply = "Aquí tienes la tendencia de tus gastos por mes."

            elif intent == "plot_pie":
                st.markdown("🥧 Here's a pie chart of your expense composition:")
                grouped = df.groupby("categoria")["valor"].sum()
                pie_chart(grouped, "Composición de gastos por categoría")
                reply = "Aquí tienes la proporción de tus gastos por categoría."

            # --- DEFAULT CHAT ---
            else:
                # Add spreadsheet context
                df_summary = f"The spreadsheet has {len(df)} rows and the following columns: {', '.join(df.columns)}."
                st.session_state.messages[0]["content"] = (
                    f"You are analyzing a financial spreadsheet named '{WORKSHEET_NAME}'. {df_summary}"
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
                st.markdown(reply)

    # Add assistant message to memory
    st.session_state.messages.append({"role": "assistant", "content": reply})
