import gradio as gr
import requests

def predict(input_data):
    response = requests.post("http://localhost:5000/predict", json=input_data)
    return response.json()

iface = gr.Interface(fn=predict, inputs="json", outputs="json")
iface.launch()