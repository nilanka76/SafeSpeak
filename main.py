from pvrecorder import PvRecorder
import speech_recognition as sr
import wave, struct
import openai


def main():
    # for index, device in enumerate(PvRecorder.get_available_devices()):
    #     print(f"[{index}] {device}")
    # idx = int(input("Enter and audio device: "))
    #
    # recorder = PvRecorder(device_index=idx, frame_length=512)  # (32 milliseconds of 16 kHz audio)
    # audio = []
    # path = 'audio_recording.wav'
    #
    # print("Started")
    # try:
    #     recorder.start()
    #     while True:
    #         frame = recorder.read()
    #         audio.extend(frame)
    # except KeyboardInterrupt:
    #     recorder.stop()
    #     with wave.open(path, 'w') as f:
    #         f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
    #         f.writeframes(struct.pack("h" * len(audio), *audio))
    # finally:
    #     print("Saved")
    #     recorder.delete()

    # openai.api_key = "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"
    # audio_file = open(r"E:\GitHub\suicidal-speech-detection\audio_recording.wav", "rb")
    # transcript = openai.Audio.transcribe("whisper-1", audio_file)
    # print(transcript)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        print("Listening... (Press Ctrl+C to stop)")
        audio = r.listen(source)
        my_text = r.recognize_google(audio)
        my_text = my_text.lower()
        print(my_text)


if __name__ == '__main__':
    main()
