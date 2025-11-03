import os
import streamlit as st
import pandas as pd
import gspread
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
SHEET_ID = st.secrets("SPREADSHEET_KEY") # 👈 replace this with your sheet ID
creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
WORKSHEET_NAME = "cashflow2"

st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

# === FUNCTION: Load Google Sheet ===
def load_google_sheet(sheet_id, worksheet_name):
    """Connect to Google Sheets and return a pandas DataFrame"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

# === Sidebar controls ===
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]
        #st.experimental_rerun()

# === Initialize chat memory ===
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# === Connect to Google Sheets ===
try:
    df = load_google_sheet(SHEET_ID, WORKSHEET_NAME)
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")
    st.dataframe(df)  # 👈 Show DataFrame before chat box
except Exception as e:
    st.error(f"Error loading Google Sheet: {e}")
    df = None

# === Initialize OpenAI ===
client = OpenAI(api_key=st.secrets("OPENAI_API_KEY"))

# === Display previous chat messages ===
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Chat input ===
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build context about the sheet
    if df is not None:
        df_summary = f"The spreadsheet has {len(df)} rows and the following columns: {', '.join(df.columns)}."
        st.session_state.messages[0]["content"] = (
            f"You are analyzing a financial spreadsheet named '{WORKSHEET_NAME}'. {df_summary}"
        )

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    # Add assistant message to memory
    st.session_state.messages.append({"role": "assistant", "content": reply})

