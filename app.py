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

broker = 'broker.emqx.io'
port = 1883
topic = "audio_test"
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

def generate_wordcloud(data):
    classification_text = ''
    for index, row in data.iterrows():
        classification_text += row['Classification'] + ' '

    if classification_text.strip() == '':
        print('Error: No text found to generate wordcloud')
        return

    wordcloud = WordCloud(width=800, height=200, background_color='white').generate(classification_text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()


def main():
    st.set_page_config(
        page_title="Real-Time Audio Classification Dashboard",
        page_icon="🎧",
        layout="wide",
    )
    st.markdown("<h1 style='text-align: center; color: black;'>🎧 Real-Time Audio Classification Dashboard</h1>", unsafe_allow_html=True)
    client = connect_mqtt()

    # creating a single-element container
    placeholder = st.empty()

    # near real-time / live feed simulation
    for seconds in range(200):
        with placeholder.container():

            # Generate wordcloud
            st.write(generate_wordcloud(df))

            # Add heatmap and pie chart below the wordcloud
            c1, c2 = st.columns((5, 5))
            with c1:
                st.markdown('### Heatmap')
                fig = go.Figure(
                    data=go.Heatmap(x=df["Timestamp"], y=df["Classification"], z=df["Accuracy"], colorscale='Viridis'))
                st.plotly_chart(fig)

            with c2:
                st.markdown('### Pie Chart')
                classifications = df['Classification'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=classifications.index, values=classifications.values)])
                st.plotly_chart(fig)
            time.sleep(1)


if __name__ == "__main__":
    main()

