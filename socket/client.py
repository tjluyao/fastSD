import socket
import gradio as gr
import io
from PIL import Image
from urllib.parse import urlparse

def communicate(text, server_url):
    parsed_url = urlparse(server_url)
    ip_address = parsed_url.hostname
    port = parsed_url.port
    print(ip_address, port)
    addr = (ip_address, int(port))
    client = socket.socket()
    client.connect(addr)
    client.send(bytes(text, encoding='gbk'))

    image_size_bytes = client.recv(4)
    image_size = int.from_bytes(image_size_bytes, 'big')

    received_size = 0
    image_data = b''
    while received_size < image_size:
        data = client.recv(4096)
        received_size += len(data)
        image_data += data
    client.close()
    return Image.open(io.BytesIO(image_data))

with gr.Blocks() as interface:
    with gr.Tab(label="tti"):
        gr.Markdown("## Optimizer Client")
        txt_in = gr.Textbox(value='一只猫', label="Input text", placeholder="Type something here...")
        url = gr.Textbox(value='http://127.0.0.1:6666/', label="Server URL", placeholder="http://")
        img_out = gr.Image(label='Output image')
        trigger = gr.Button('Send request')
        trigger.click(communicate, inputs=[txt_in, url], outputs=img_out)

if __name__ == "__main__":
    interface.launch()