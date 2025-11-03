import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json

st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

WORKSHEET_NAME = "cashflow2"

# --- Load Google Sheets ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_google_sheets():
    try:
        # Load credentials directly from Streamlit secrets
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

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)

        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_google_sheets()
if error:
    st.error(f"Error loading Google Sheet: {error}")
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")
    st.dataframe(df)  # Show the DataFrame above chat

# --- Initialize chat memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# --- Sidebar options ---
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# --- Initialize OpenAI ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Display previous messages ---
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add spreadsheet summary context
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


