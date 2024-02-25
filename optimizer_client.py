import gradio as gr
import requests
import io
from PIL import Image

def communicate(text, server_url):
    response = requests.post(server_url, json={'prompt': text})
    image = response.content
    return Image.open(io.BytesIO(image))

with gr.Blocks() as interface:
    with gr.Tab(label="tti"):
        gr.Markdown("## Optimizer Client")
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_out = gr.Image(label='Output image')
        trigger = gr.Button('Send request')
        trigger.click(communicate, inputs=[txt_in, url], outputs=img_out)


if __name__ == "__main__":
    interface.launch()