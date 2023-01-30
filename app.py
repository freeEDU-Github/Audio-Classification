#For testing purpose only

import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
page_title="Real-Time Audio Classification Dashboard",
page_icon="🎧",
layout="wide",
)

dataset_url = "classification_log.csv"

@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

st.title("Real-Time Audio Classification Dashboard")

#top-level filters
classification_filter = st.selectbox("Select the Classification", pd.unique(df["Classification"]))

#creating a single-element container
placeholder = st.empty()

#dataframe filter
df = df[df["Classification"] == classification_filter]

for seconds in range(200):
#creating KPIs
    avg_accuracy = round(np.mean(df["Accuracy"]), 2)
with placeholder.container():
    kpi = st.columns(1)

    kpi[0].metric(
        label="Accuracy",
        value=avg_accuracy,
        delta=round(avg_accuracy - 0.5, 2),
    )

st.markdown("### Detailed Data View")
st.dataframe(df)
time.sleep(1)