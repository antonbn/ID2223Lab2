from transformers import pipeline
from pytube import YouTube
import gradio as gr

pipe = pipeline(model="id2223lab1/whisper-small")


def transcribe(url):
	audio = YouTube(url).streams.filter(file_extension='mp4', only_audio=True).first().download()

	text = pipe(audio, batch_size=512, truncation=True)["text"]

	return text


iface = gr.Interface(
	fn=transcribe,
	inputs=gr.Textbox(label="Enter a YouTube URL:"),
	outputs="text",
	title="Whisper Small SE",
	description="Transcribe swedish videos",
)

iface.launch()
