import streamlit as st
import pandas as pd
import gspread
from google.oauth2 import service_account
from util import execute_intent, extract_intent

# ----------------------------
#  PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Financial Chatbot with Google Sheets", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

# ----------------------------
#  LOAD GOOGLE SHEET
# ----------------------------
@st.cache_data(ttl=300)
def load_sheet(sheet_name="cashflow2"):
    """Load spreadsheet data from Google Sheets using Streamlit Secrets."""
    try:
        # ✅ Credentials from Streamlit secrets
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"], scopes=scopes
        )
        client = gspread.authorize(creds)

        # ✅ Spreadsheet key also from secrets
        spreadsheet = client.open_by_key(st.secrets["SPREADSHEET_KEY"])
        sheet = spreadsheet.worksheet(sheet_name)
        data = sheet.get_all_records()

        # ✅ Convert to DataFrame
        df = pd.DataFrame(data)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')

        if 'categoria' in df.columns:
            df['categoria'] = df['categoria'].astype(str).str.lower()
        if 'detalle' in df.columns:
            df['detalle'] = df['detalle'].astype(str).str.lower()
        if 'mes' not in df.columns:
            df['mes'] = df['fecha'].dt.month_name(locale='es_CO').str.lower()
        if 'dow' not in df.columns:
            df['dow'] = df['fecha'].dt.day_name(locale='es_CO').str.lower()

        return df
    except Exception as e:
        st.error(f"❌ Failed to load Google Sheet: {e}")
        return None

# Load data once
df = load_sheet()

if df is not None:
    st.success("✅ Google Sheet 'cashflow2' loaded successfully!")
    st.dataframe(df.head(10))
else:
    st.stop()

# ----------------------------
#  CHAT INTERFACE
# ----------------------------
st.subheader("💬 Chat with your financial data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask something about your expenses (e.g. 'cuánto gasté en taxis en septiembre')")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # --- Extract the intent from the message ---
    intent = extract_intent(user_input)

    # --- Execute intent and get result ---
    result = execute_intent(df, intent)

    # --- Show assistant message ---
    if "error" in result:
        response = f"⚠️ {result['error']}"
    elif "result" in result:
        response = f"💡 {result['result']}"
    else:
        response = "🤔 I couldn't find a clear answer."

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# ----------------------------
#  DISPLAY CHAT
# ----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
