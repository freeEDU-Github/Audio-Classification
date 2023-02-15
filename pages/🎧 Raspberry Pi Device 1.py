import random
import sys
import time
import plotly.graph_objects as go
import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import json

from matplotlib import pyplot as plt
from wordcloud import WordCloud

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
        page_title="Real-Time Audio Classification Dashboard",
        page_icon="🎧",
        layout="wide",
    )
st.markdown("<h1 style='text-align: center; color: black;'>🎧 Raspberry Pi Device 1 Dashboard</h1>",
                unsafe_allow_html=True)

broker = 'broker.emqx.io'
port = 1883
topic = "device1"
client_id = f'python-mqtt-{random.randint(0, 100)}'
username = 'emqx'
password = 'public'

def connect_mqtt() -> mqtt:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(topic)
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    client.loop_start()
    return client

df = pd.DataFrame(columns=["Timestamp", "Classification", "Accuracy"])

def on_message(client, userdata, msg):
    topic = msg.topic
    message = msg.payload.decode()
    data_json = json.loads(message)
    global df
    df = df.append(data_json, ignore_index=True)
    print("Data: ", df)
    print("Classification: ", data_json["Classification"])

state = df

wordcloud_image = st.empty()
columns = st.columns((5, 5))
heatmap = columns[0].empty()
pie_chart = columns[1].empty()

def update_plot():
    global df
    global state
    global wordcloud_image
    global heatmap
    global pie_chart

    if df.equals(state):
        return

    state = df.copy()
    with st.spinner("Generating wordcloud image..."):
        classification_text = ''
        for index, row in state.iterrows():
            classification_text += row['Classification'] + ' '

        if classification_text.strip() == '':
            print('Error: No text found to generate wordcloud')
            return

        wordcloud = WordCloud(width=800, height=200, background_color='white').generate(classification_text)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        wordcloud_image.pyplot()

    with st.spinner("Generating heatmap..."):
        fig = go.Figure(
            data=go.Heatmap(x=state["Timestamp"], y=state["Classification"], z=state["Accuracy"], type='heatmap')
        )
        fig.update_layout(title='Accuracy over time', xaxis_title='Timestamp', yaxis_title='Classification')
        heatmap.plotly_chart(fig)

    with st.spinner("Generating pie chart..."):
        count = state["Classification"].value_counts()
        labels = count.index
        values = count.values
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values)]
        )
        fig.update_layout(title='Classification distribution')
        pie_chart.plotly_chart(fig)

def main():
    client = connect_mqtt()
    while True:
        update_plot()
        time.sleep(1)

if __name__ == "__main__":
    main()