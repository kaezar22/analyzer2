import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json
import re

from utils import summarize_expense, format_money

st.set_page_config(page_title="Cashflow Analyst", page_icon="💬", layout="wide")
st.title("💬 Cashflow Analyst — Chat with your data")

WORKSHEET_NAME = "cashflow2"

# ------------------------------------------------------------------
# Load Google Sheets
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
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

        # Try to convert numeric fields safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_google_sheets()
if error:
    st.error(f"Error loading Google Sheet: {error}")
    st.stop()
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")

# ------------------------------------------------------------------
# Chat memory and sidebar
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.experimental_rerun()

st.write("Ask questions about your cashflow 💰")
user_input = st.chat_input("Ejemplo: ¿Cuánto gasté en taxis en octubre?")

# ------------------------------------------------------------------
# Display previous chat
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Chat logic
# ------------------------------------------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------------------------------------------------------
    # 1️⃣ Simple pattern detection (category + month)
    # ------------------------------------------------------------------
    match = re.search(r"en (\w+)", user_input.lower())
    month = match.group(1) if match else None

    categories = ["taxis", "cerveza", "comida", "transporte", "supermercado"]
    category = None
    for c in categories:
        if c in user_input.lower():
            category = c
            break

    if df is not None and category:
        try:
            result = summarize_expense(df, category=category, month=month)
            reply = result["respuesta"]
        except Exception as e:
            reply = f"Error calculando el resultado: {e}"
    else:
        # ------------------------------------------------------------------
        # 2️⃣ If no direct pattern, fall back to OpenAI reasoning
        # ------------------------------------------------------------------
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        context = f"You are a financial analyst. The dataframe columns are: {', '.join(df.columns)}"
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": user_input}
        ]
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        reply = response.choices[0].message.content

    # ------------------------------------------------------------------
    # 3️⃣ Show assistant reply
    # ------------------------------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
