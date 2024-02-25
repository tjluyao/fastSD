### 采用Threading的方式运行后端服务，同时启动Flask服务，接收前端请求；不知是否可以改善runtime成为异步函数，以便在Flask中调用
from optimizer_sd import sd_optimizer, sd_request
from io import BytesIO
from flask import Flask, request, send_file
import asyncio
from PIL import Image
import numpy as np
import threading

class sd_server(sd_optimizer):
    def __init__(self, config_file):
        super().__init__(config_file)

    def runtime(self):
        while True:
            for i,waitlist in enumerate(self.waitlists):
                if len(waitlist) == 0:
                    continue

                batch_size = self.batch_configs[i]
                batch = self.select(waitlist,batch_size)
                for item in batch:
                    waitlist.remove(item)
                
                if i == 0:
                    self.preprocess(batch)
                elif i == 1:
                    self.iteration(batch)
                elif i == 2:
                    self.postprocess(batch)

                for item in batch:
                    if isinstance(item.state,int):
                        self.waitlists[item.state].append(item)
                    else:
                        pass

    def respond(self, text):
        req = sd_request(
                        state=self.state,
                        prompt=text,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
        self.waitlists[0].append(req)

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
async def handle_request():
    if request.method == 'POST':

        data = request.get_json()  
        req = sd_request(
                        state=optimizer.state,
                        prompt=data['prompt'],
                        video_task=False,
                        num_samples=1,
                    )
        optimizer.waitlists[0].append(req)
        print('Request added')
        while req.output is None:
            await asyncio.sleep(1)
            print('waiting for result...')
        
        img = Image.fromarray(np.uint8(req.output))
        image_stream = BytesIO()
        img.save(image_stream, format='JPEG')
        image_stream.seek(0)
        print('Request sending')
        return send_file(image_stream, mimetype='image/jpeg')
    
    elif request.method == 'GET':
        return 'Received GET request'

if __name__ == '__main__':
    optimizer = sd_server(config_file='configs/sd_21_512.yaml')
    backend_thread = threading.Thread(target=optimizer.runtime, daemon=True)
    backend_thread.start()
    app.run()   # Start the server