import gradio as gr
import requests
import io
import numpy as np
from PIL import Image

def tti_conn(model_name, text, server_url):
    response = requests.post(server_url, json={'model_name': model_name[0], 'prompt': text})
    image = response.content
    return Image.open(io.BytesIO(image))

def itv_conn(model_name, text, server_url, img_in):
    img_in = img_in.tolist()
    response = requests.post(server_url, json={'model_name': model_name, 'prompt': text, 'img_in': img_in})
    image = response.content
    return Image.open(io.BytesIO(image))

with gr.Blocks() as interface:
    gr.Markdown("## Optimizer Client for Stable Diffusion")
    gr.Markdown("This is a client for the stable diffusion model. It can be used to generate images from text prompts.")
    with gr.Tab(label="tti"):
        model_name = gr.CheckboxGroup(label="Model", choices=["stable_diffusion xl", "stable_diffusion 2.1"])
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_out = gr.Image(label='Output image')
        trigger = gr.Button('Send request')
        trigger.click(tti_conn, inputs=[model_name, txt_in, url], outputs=img_out)
    
    with gr.Tab(label="itv"):
        model_name = gr.Label("Stable Video Diffusion")
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_in = gr.Image(label='Input image', sources='upload')
        img_out = gr.Image(label='Output image')
        #video_out = gr.Video(label='Output video')
        trigger = gr.Button('Send request')
        trigger.click(itv_conn, inputs=[model_name, txt_in, url, img_in], outputs=img_out)
    
    gr.Markdown("Author: 沈骏一")


if __name__ == "__main__":
    interface.launch()