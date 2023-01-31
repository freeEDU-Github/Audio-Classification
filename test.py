import time

import numpy as np
import pandas as pd
import streamlit as st

data = pd.read_csv('classification_log.csv')

st.set_page_config(
page_title="Real-Time Audio Classification Dashboard",
page_icon="🎧",
layout="wide",
)

def search(data, column, search_term):
    if column == 'Classification':
        search_term = int(search_term)
    indexes = data.loc[data[column].isin([search_term])].index
    if indexes.size > 0:
        return data.iloc[indexes]
    else:
        return []