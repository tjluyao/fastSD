import gradio as gr
import requests
import io
import numpy as np
from PIL import Image

def tti_conn(model_name, text, server_url):
    response = requests.post(server_url, json={'model_name': model_name[0], 'prompt': text})
    image = response.content
    return Image.open(io.BytesIO(image))

def iti_conn(model_name, text, server_url, img_in):
    response = requests.post(server_url, json={'model_name': model_name[0], 'prompt': text, 'img_in': img_in.tolist()})
    image = response.content
    return Image.open(io.BytesIO(image))

def itv_conn(model_name, text, server_url, img_in):
    response = requests.post(server_url, json={'model_name': model_name, 'prompt': text, 'img_in': img_in.tolist()})
    #image = response.content
    #return Image.open(io.BytesIO(image))
    video_bytes = response.content
    with open('temp/temp_out.mp4', 'wb') as f:
        f.write(video_bytes)
    return 'temp_out.mp4'

def download_img(img, path):
    img = Image.fromarray(img)
    img.save(path)

def download_video(video, path):
    with open(path, 'wb') as f:
        f.write(video)

with gr.Blocks() as interface:
    gr.Markdown("## Optimizer Client for Stable Diffusion")
    gr.Markdown("This is a client for the stable diffusion model. It can be used to generate images from text prompts.")
    with gr.Tab(label="tti"):
        gr.Markdown("Input: Text prompt of image description.")
        gr.Markdown("Output: Generated Image Sample.")
        model_name = gr.CheckboxGroup(label="Model", choices=["stable_diffusion xl", 
                                                              "stable_diffusion 2.1",
                                                              "stable_diffusion xl-turbo",
                                                              "stable_diffusion-turbo"
                                                              ])
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_out = gr.Image(label='Output image')
        down_pth = gr.Textbox(value='output.jpg', label="Download Path", placeholder="output.jpg")
        with gr.Row():
            trigger = gr.Button('Send request')
            clearner = gr.ClearButton([img_out])
            downloader = gr.Button('Download')
        trigger.click(tti_conn, inputs=[model_name, txt_in, url], outputs=img_out)
        downloader.click(download_img, inputs=[img_out, down_pth])


    with gr.Tab(label="iti"):
        gr.Markdown("Input: Image Sample using to generate.")
        gr.Markdown("Output: Generated Image")
        model_name = gr.CheckboxGroup(label="Model", choices=["stable_diffusion xl", 
                                                              "stable_diffusion 2.1",
                                                              ])
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        img_in = gr.Image(label='Input image', sources='upload')
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_out = gr.Image(label='Output image')
        down_pth = gr.Textbox(value='output.jpg', label="Download Path", placeholder="output.jpg")
        with gr.Row():
            trigger = gr.Button('Send request')
            clearner = gr.ClearButton([img_out])
            downloader = gr.Button('Download')
        trigger.click(iti_conn, inputs=[model_name, txt_in, url, img_in], outputs=img_out)
        downloader.click(download_img, inputs=[img_out, down_pth])
    
    with gr.Tab(label="itv"):
        model_name = gr.Label("Stable Video Diffusion")
        gr.Markdown("Input: Image sample of video generation.")
        gr.Markdown("Output: Generated Video")
        txt_in = gr.Textbox(value='A cat', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:5000/', label="Server URL", placeholder="http://")
        img_in = gr.Image(label='Input image', sources='upload')
        video_out = gr.Video(label='Output video')
        down_pth = gr.Textbox(value='output.mp4', label="Download Path", placeholder="output.mp4")
        with gr.Row():
            trigger = gr.Button('Send request')
            clearner = gr.ClearButton([video_out])
            downloader = gr.Button('Download')
        trigger.click(itv_conn, inputs=[model_name, txt_in, url, img_in], outputs=video_out)
        downloader.click(download_video, inputs=[video_out, down_pth])
    
    gr.Markdown("Author: 沈骏一")
    gr.Markdown("College of Control Science and Engineering, Zhejiang University")


if __name__ == "__main__":
    interface.launch(
        share=True,
    )