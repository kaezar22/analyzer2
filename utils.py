# utils.py
from typing import Optional
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -- helper to format money
def format_money(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# -- line chart for trends (expects df with a datetime-like index or a column)
def line_chart(df: pd.DataFrame, x: str, y: str, title: Optional[str] = None):
    """
    Draws a line chart (matplotlib) and displays in Streamlit.
    df: dataframe
    x: column name for x (datetime recommended)
    y: column name for numeric values
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df[x], df[y], marker='o', linewidth=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

# -- bar chart for category breakdowns
def bar_chart(df: pd.DataFrame, x: str, y: str, title: Optional[str] = None, rotate_xticks: bool = True):
    """
    df grouped by x with aggregated y (numeric)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x].astype(str), df[y])
    if rotate_xticks:
        plt.xticks(rotation=45, ha='right')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    plt.close(fig)

# -- pie chart for composition
def pie_chart(series: pd.Series, title: Optional[str] = None):
    """
    series: index are labels, values are numeric (can include negatives)
    """
    # Convert to absolute values (since expenses are negative)
    series = series.abs()

    # Remove zero values (no wedge)
    series = series[series > 0]

    if series.empty:
        st.warning("No data available for pie chart.")
        return

    # Create chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(series.values, labels=series.index.astype(str), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    if title:
        ax.set_title(title)

    st.pyplot(fig)
    plt.close(fig)


# -- small helper to display a dataframe prettily in streamlit
def display_table(df: pd.DataFrame, n: int = 20):
    st.dataframe(df.head(n))

# -- helper to aggregate by timeframe (month, week, dow)
def aggregate_by_timeframe(df: pd.DataFrame, timeframe: str = "mes", value_col: str = "valor"):
    """
    timeframe: 'mes', 'dow', or 'week' (week number)
    returns grouped sums
    """
    if timeframe == "mes":
        grouped = df.groupby('mes')[value_col].sum().sort_index()
    elif timeframe == "dow":
        grouped = df.groupby('dow')[value_col].sum().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], fill_value=0)
    elif timeframe == "week":
        if 'week' not in df.columns:
            df['week'] = df['fecha'].dt.isocalendar().week
        grouped = df.groupby('week')[value_col].sum().sort_index()
    else:
        grouped = df.groupby(timeframe)[value_col].sum()
    return grouped
