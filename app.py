import streamlit as st
from tabs import overview, patterns, insights, time
st.set_page_config(page_title="Atlas", page_icon="🛒", layout="wide")

st.sidebar.image("assets/logo.png", width=160)
st.sidebar.title("Atlas Navigation")

tab = st.sidebar.radio(
    "Select a view:",
    ["Overview", "Basket Patterns", "Time Dynamics", "Insights"]
)
if tab == "Overview":
    overview.show()
elif tab == "Basket Patterns":
    patterns.show()
elif tab == "Time Dynamics":
    time.show()
else:
    insights.show()
