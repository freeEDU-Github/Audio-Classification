import argparse
import time

from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor

import tkinter
import tkinter as tk
from tkinter import ttk
import pandas as pd
import plotly.express as px
import csv
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from wordcloud import WordCloud

root = Tk()
root.geometry("600x660")

min_prediction = 50

def logs():
    root = tk.Tk()
    root.title("Audio Classification Heatmap")

    # read classification_log.csv file
    df = pd.read_csv("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classification_log.csv")

    # create a tab control
    tab_control = ttk.Notebook(root)

    # create a tab for the heatmap
    heatmap_tab = ttk.Frame(tab_control)
    tab_control.add(heatmap_tab, text="Heatmap")

    # create a heatmap using plotly
    fig = px.density_heatmap(data_frame=df, y="Timestamp", x="Classification")

    # display the heatmap on the tab
    fig.show()

    tab_control.pack(expand=1, fill="both")
    root.mainloop()


def wordcloud_tab():
    root = tk.Tk()
    root.title("Audio Classification Wordcloud")

    # read classification_log.csv file
    df = pd.read_csv("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classification_log.csv")

    # create a tab control
    tab_control = ttk.Notebook(root)

    # create a tab for the wordcloud
    wordcloud_tab = ttk.Frame(tab_control)
    tab_control.add(wordcloud_tab, text="Wordcloud")

    # create a wordcloud using the classification column from the dataset
    wc = WordCloud(background_color='white', width=800, height=400).generate(' '.join(df['Classification']))

    # display the wordcloud on the tab
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    tab_control.pack(expand=1, fill="both")
    root.mainloop()


rectangle = tkinter.Canvas(root, width=600, height=50)
rectangle.place(x=0, y=0)

button = tkinter.Button(root, borderwidth=3, width=62, height=1, text="Heatmap", command=lambda: logs())
button.place(relx=0.5, rely=0.92, anchor='s')

button = tkinter.Button(root, borderwidth=3, width=62, height=1, text="Word Cloud", command=lambda: wordcloud_tab())
button.place(relx=0.5, rely=0.98, anchor='s')

slider = tkinter.Scale(root, from_=1, to=60, orient='horizontal', resolution=1, length=590, tickinterval=10)
slider.place(relx=0.5, rely=0.68, anchor='s')
label = tkinter.Label(root, text='Prediction time interval in seconds')
label.place(in_=slider, relx=0.5, rely=1.4, anchor="s")


def set_min_prediction(value):
    global min_prediction
    min_prediction = int(value)


min_prediction_scale = tkinter.Scale(root, from_=0, to=100, orient='horizontal', resolution=1,
                                     sliderlength=20, length=590, showvalue=True, tickinterval=10,
                                     command=lambda event: set_min_prediction(event))
min_prediction_scale.place(relx=0.5, rely=0.82, anchor='s')
label = tkinter.Label(root, text='Minimum prediction percentage')
label.place(in_=slider, relx=0.5, rely=3, anchor="s")
min_prediction_scale.set(50)


def graphic_ui(result):
    label1 = tkinter.Label(root)
    label1.place(x=5, y=50)
    Label2 = tkinter.Label(root, bg="#6400FF", bd=91, height=8, width=12)
    Label2.place(x=300, y=50)
    label3 = tkinter.Label(root, bg="#6400FF", fg='white', font="Castellar, 50")
    label3.place(relx=0.75, rely=0.38, anchor='center')
    label4 = tkinter.Label(root, bg="#6400FF", fg='white', font="Castellar, 20")
    label4.place(relx=0.75, rely=0.18, anchor='center')

    def update_ui():
        classification = result.classifications[0]
        label_list = [category.category_name for category in classification.categories]
        score_list = [category.score for category in classification.categories]
        percentage = "{:.0%}".format(score_list[0])
        
        if label_list[0] == 'Police car (siren)' or label_list[0] == 'Civil defense siren' or label_list[0] == 'Smoke alarm' or label_list[0] == 'Screaming' or label_list[0] == 'Shout' or label_list[0] == 'Siren':
            rectangle.create_rectangle(0, 0, 600, 50, fill="red")
            rectangle.create_text(300, 25, text="Danger", font=("Castellar", 20), fill="white")
        elif label_list[0] == 'Bus' or label_list[0] == 'Truck' or label_list[0] == 'Train' or label_list[0] == 'Inside, public space' or label_list[0] == 'Applause':
            rectangle.create_rectangle(0, 0, 600, 50, fill="yellow")
            rectangle.create_text(300, 25, text="Risky", font=("Castellar", 20), fill="white")
        elif label_list[0] == 'Silence' or label_list[0] == 'Typing' or label_list[0] == 'Printer' or label_list[0] == 'Computer keyboard' or label_list[0] == 'Snoring':
            rectangle.create_rectangle(0, 0, 600, 50, fill="green")
            rectangle.create_text(300, 25, text="Safe", font=("Castellar", 20), fill="white")
        else:
            rectangle.create_rectangle(0, 0, 600, 50, fill="white")

        #Safe
        if label_list[0] == 'Silence':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/silence.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Typing':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/keyboard_typing.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Printer':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Printer.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Computer keyboard':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Computer keyboard.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Snoring':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Snoring.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test

        #Risky
        elif label_list[0] == 'Bus':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Bus.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Truck':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Truck.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Train':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Train.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Inside, public space':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Inside, public space.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Applause':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Applause.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test

        #Danger
        elif label_list[0] == 'Police car (siren)':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Police car (siren).jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Civil defense siren' or label_list[0] == 'Siren':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Civil defense siren.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Smoke alarm':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Smoke alarm.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Screaming':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Screaming.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Shout':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Shout.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test


        elif label_list[0] == 'Speech':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/talking.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Animal':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/dog.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Synthesizer' or label_list[0] == 'Music' or label_list[0] == 'Keyboard (musical)':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/synth")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Car Alarm':
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/car_alarm.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'DoorBell':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/doorbelljpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Guitar':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/guitar.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Conversation':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Conversation.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Alarm clock':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Alarm clock.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Radio':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Radio.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        elif label_list[0] == 'Christmas music':
            image1 = Image.open("/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/Christmas music.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        
        else:
            image1 = Image.open(
                "/home/pi/examples/lite/examples/audio_classification/raspberry_pi/classes/none.jpg")
            img = image1.resize((295, 320))
            test = ImageTk.PhotoImage(img)
            label1.configure(image=test)
            label1.image = test
        label3.configure(text=percentage)
        label4.configure(text=label_list[0])
        label3.lift()
        label4.lift()

    root.after(1000, update_ui)
    root.update()


# Open the CSV file for writing
log_file = open("classification_log.csv", "w", newline='')
fieldnames = ['Timestamp', 'Classification', 'Accuracy']
writer = csv.DictWriter(log_file, fieldnames=fieldnames)
writer.writeheader()


def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float, num_threads: int,
        enable_edgetpu: bool) -> None:
    if (overlapping_factor <= 0) or (overlapping_factor >= 1.0):
        raise ValueError('Overlapping factor must be between 0 and 1.')

    if (score_threshold < 0) or (score_threshold > 1.0):
        raise ValueError('Score threshold must be between (inclusive) 0 and 1.')

    # Initialize the audio classification model.
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    classification_options = processor.ClassificationOptions(
        max_results=max_results, score_threshold=score_threshold)
    options = audio.AudioClassifierOptions(
        base_options=base_options, classification_options=classification_options)
    classifier = audio.AudioClassifier.create_from_options(options)

    audio_record = classifier.create_audio_record()
    tensor_audio = classifier.create_input_tensor_audio()

    input_length_in_second = float(len(
        tensor_audio.buffer)) / tensor_audio.format.sample_rate
    interval_between_inference = input_length_in_second * (1 - overlapping_factor)
    pause_time = interval_between_inference * 0.1
    last_inference_time = time.time()

    ui_val = graphic_ui

    audio_record.start_recording()

    while True:
        now = time.time()
        diff = now - last_inference_time
        if diff < interval_between_inference:
            time.sleep(pause_time)
            continue
        last_inference_time = now

        tensor_audio.load_from_audio_record(audio_record)
        result = classifier.classify(tensor_audio)

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        category = result.classifications[0].categories[0]
        score = round(category.score * 100, 2)
        if score >= min_prediction:
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            writer.writerow({'Timestamp': time_stamp, 'Classification': category.category_name, 'Accuracy': score})
            log_file.flush()
        time_interval = slider.get()
        time.sleep(time_interval)
        ui_val(result)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the audio classification model.',
        required=False,
        default='yamnet.tflite')
    parser.add_argument(
        '--maxResults',
        help='Maximum number of results to show.',
        required=False,
        default=1)
    parser.add_argument(
        '--overlappingFactor',
        help='Target overlapping between adjacent inferences. Value must be in (0, 1)',
        required=False,
        default=0.5)
    parser.add_argument(
        '--scoreThreshold',
        help='The score threshold of classification results.',
        required=False,
        default=0.0)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    run(args.model, int(args.maxResults), float(args.scoreThreshold),
        float(args.overlappingFactor), int(args.numThreads),
        bool(args.enableEdgeTPU))


if __name__ == '__main__':
    main()
