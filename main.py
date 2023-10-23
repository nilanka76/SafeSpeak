# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import speech_recognition as sr
#
# MODEL_SAVE_PATH = r"E:\GitHub\suicidal-speech-detection\distilbert_base_uncased"
#
# # def save_audio(filename, audio_data):
# #     with open(filename, "wb") as f:
# #         f.write(audio_data)
#
#
# def main():
#     r = sr.Recognizer()
#     my_text = ""
#
#     # with sr.AudioFile("recorded_audio.wav") as source:
#     #     audio = r.record(source)
#
#     with sr.Microphone() as source:
#         try:
#             r.adjust_for_ambient_noise(source, duration=0.3)
#             print("Listening...")
#             audio = r.listen(source, timeout=3)  # after 3 second of silence it will stop recording.
#             my_text = r.recognize_google(audio)
#             print(my_text)
#             # save_audio("recorded_audio.wav", audio.get_wav_data())
#         except sr.WaitTimeoutError:
#             print("Ended...")
#
#     if my_text == "":
#         print("No speech detected")
#         return
#
#     # Load model
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
#
#     inputs = tokenizer(my_text, return_tensors="pt", padding=True, truncation=True)
#
#     # Make the prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     predicted_probabilities = torch.sigmoid(outputs.logits)
#
#     predicted_probability_positive_class = predicted_probabilities[0][1].item()
#     predicted_probability_negative_class = predicted_probabilities[0][0].item()
#
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     if predicted_label == 1:
#         predicted_class = "Suicidal"
#         predicted_probability = predicted_probability_positive_class
#     else:
#         predicted_class = "Not Suicidal"
#         predicted_probability = predicted_probability_negative_class
#
#     print(f"Speaker is {predicted_probability*100:.2f}% {predicted_class}")
#
#
# if __name__ == '__main__':
#     main()

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import speech_recognition as sr
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_SAVE_PATH = r"E:\GitHub\suicidal-speech-detection\distilbert_base_uncased"


def predict_speech(text):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_probabilities = torch.sigmoid(outputs.logits)

    predicted_probability_positive_class = predicted_probabilities[0][1].item()
    predicted_probability_negative_class = predicted_probabilities[0][0].item()

    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    if predicted_label == 1:
        predicted_class = "Suicidal"
        predicted_probability = predicted_probability_positive_class
    else:
        predicted_class = "Not Suicidal"
        predicted_probability = predicted_probability_negative_class

    return f"Speaker is {predicted_probability*100:.2f}% {predicted_class}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        r = sr.Recognizer()
        my_text = ""

        with sr.Microphone() as source:
            try:
                r.adjust_for_ambient_noise(source, duration=0.3)
                print("Listening...")
                audio = r.listen(source, timeout=3)  # after 3 seconds of silence, it will stop recording.
                my_text = r.recognize_google(audio)
                print(my_text)
            except sr.UnknownValueError:
                return render_template("index.html", text="Speech not recognized", predicted_text="")
            except sr.WaitTimeoutError:
                print("Ended...")

        if my_text == "":
            return render_template("index.html", text="Please click the button and start speaking", predicted_text="")
        else:
            predicted_text = predict_speech(my_text)
            return render_template("index.html", text=my_text, predicted_text=predicted_text)

    return render_template("index.html", predicted_text="", predicted_text_id="predicted-text")


if __name__ == '__main__':
    app.run(debug=True)