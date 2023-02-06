import random
import streamlit as st
import wordcloud
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json

broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'emqx'
password = 'public'

def connect_mqtt() -> mqtt:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    client.loop_start()
    return client

def get_data_from_mqtt(client):
    received_data = []

    # Callback when a message is published to the specified topic.
    def on_message(client, userdata, msg):
        print(msg.topic+" "+str(msg.payload))
        received_data.append(json.loads(msg.payload))

    client.subscribe(topic)
    client.on_message = on_message

    # Continuously listen for incoming messages.
    client.loop_start()
    return received_data

def create_wordcloud(text):
    wc = wordcloud.WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot()

def main():
    client = connect_mqtt()
    data = get_data_from_mqtt(client)
    text = ' '.join([d['Classification'] for d in data])
    if len(data) == 0 or text == '':
        st.write("No data to display wordcloud")
    else:
        create_wordcloud(text)

    create_wordcloud(text)

if __name__ == '__main__':
    main()
