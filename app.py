import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import json
from utils import line_chart, bar_chart, pie_chart, display_table, aggregate_by_timeframe

st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Financial Chatbot with Google Sheets")

WORKSHEET_NAME = "cashflow2"

# --- Load Google Sheets ---
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

        # Parse numeric/date columns
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        if "valor" in df.columns:
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_google_sheets()
if error:
    st.error(f"❌ Error loading Google Sheet: {error}")
    st.stop()
else:
    st.success(f"✅ Google Sheet '{WORKSHEET_NAME}' loaded successfully!")
    st.dataframe(df)

# --- Initialize OpenAI ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Chat memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "You are a financial assistant analyzing a Google Sheet called 'cashflow2'. "
            "It contains columns: fecha (date), categoria (category), detalle (sub-category), "
            "valor (amount), medio (payment method), mes (month), dow (day of week). "
            "'ingreso' represents income; all other categories are expenses. "
            "You can call plotting functions to visualize data when requested."
        )}
    ]

# --- Sidebar options ---
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.experimental_rerun()

# --- Display previous messages ---
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Define available tools ---
def tool_line_chart(x_col: str, y_col: str):
    line_chart(df, x_col, y_col, title=f"{y_col} over {x_col}")

def tool_bar_chart(x_col: str, y_col: str):
    bar_chart(df.groupby(x_col)[y_col].sum().reset_index(), x_col, y_col, title=f"{y_col} by {x_col}")

def tool_pie_chart(group_col: str):
    pie_chart(df.groupby(group_col)["valor"].sum(), title=f"Composition by {group_col}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "tool_line_chart",
            "description": "Draws a line chart with given x and y columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_col": {"type": "string", "description": "Column for x-axis"},
                    "y_col": {"type": "string", "description": "Column for y-axis"}
                },
                "required": ["x_col", "y_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_bar_chart",
            "description": "Draws a bar chart grouping by x_col, summing y_col.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_col": {"type": "string"},
                    "y_col": {"type": "string"}
                },
                "required": ["x_col", "y_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_pie_chart",
            "description": "Draws a pie chart showing value composition by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string"}
                },
                "required": ["group_col"]
            }
        }
    },
]

# --- Chat input ---
user_input = st.chat_input("Ask something about your cashflow...")

if user_input:
    # Add user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Get assistant response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                tools=tools,
            )

            choice = response.choices[0]
            msg = choice.message

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    func_name = call.function.name
                    args = json.loads(call.function.arguments)

                    st.markdown(f"🛠 Using tool: `{func_name}` with args `{args}`")

                    # Execute the requested tool
                    if func_name == "tool_line_chart":
                        tool_line_chart(**args)
                    elif func_name == "tool_bar_chart":
                        tool_bar_chart(**args)
                    elif func_name == "tool_pie_chart":
                        tool_pie_chart(**args)
                    else:
                        st.warning(f"Unknown tool: {func_name}")
                reply = "Here's the requested chart."
            else:
                reply = msg.content
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
