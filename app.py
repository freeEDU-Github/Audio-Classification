import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# Page Setting
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
page_title="Real-Time Audio Classification Dashboard",
page_icon="🎧",
layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Audio Data
data = 'classification_log.csv'
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(data)

df = get_data()

st.markdown("<h1 style='text-align: center; color: black;'>🎧 Real-Time Audio Classification Dashboard</h1>", unsafe_allow_html=True)

#Word Cloud
accuracies = df["Accuracy"]

wordcloud = WordCloud(width = 800, height = 200,
                background_color ='white',
                min_font_size = 10).generate(' '.join(df['Classification']))

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

st.pyplot()

# Create 2 columns for Heatmap and Pie Chart

c1, c2 = st.columns((5,5))
with c1:
    st.markdown('### Heatmap')
    fig = go.Figure(data=go.Heatmap(x=df["Timestamp"], y=df["Classification"], z=df["Accuracy"], colorscale='Viridis'))
    st.plotly_chart(fig)

with c2:
    st.markdown('### Pie Chart')
    classifications = df['Classification'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=classifications.index, values=classifications.values)])
    st.plotly_chart(fig)