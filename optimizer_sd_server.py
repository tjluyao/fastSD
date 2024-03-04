### 采用Threading的方式运行后端服务，同时启动Flask服务，接收前端请求；不知是否可以改善runtime成为异步函数，以便在Flask中调用
from optimizer_sd import sd_optimizer, sd_request
from io import BytesIO
from flask import Flask, request, send_file
import asyncio
from PIL import Image
import numpy as np
import threading, multiprocessing

class sd_server():
    def __init__(self, config_files):
        self.optimizers = []
        for file in config_files:
            optimizer = sd_optimizer(file)
            self.optimizers.append(optimizer)

    def runtime(self):
        while True:
            for optimizer in self.optimizers:
                for i,waitlist in enumerate(optimizer.waitlists):
                    if len(waitlist) == 0:
                        continue

                    batch_size = optimizer.batch_configs[i]
                    batch = optimizer.select(waitlist,batch_size)
                    for item in batch:
                        waitlist.remove(item)
                    
                    if i == 0:
                        optimizer.preprocess(batch)
                    elif i == 1:
                        optimizer.iteration(batch)
                    elif i == 2:
                        optimizer.postprocess(batch)

                    for item in batch:
                        if isinstance(item.state,int):
                            optimizer.waitlists[item.state].append(item)
                        else:
                            pass

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
async def handle_request():
    if request.method == 'POST':
        data = request.get_json()  
        if data['model_name'] in ['stable_diffusion 2.1',
                                  'stable_diffusion xl',
                                  'stable_diffusion xl-turbo',
                                  'stable_diffusion-turbo']:
            if data['model_name'] == 'stable_diffusion 2.1':
                optimizer = server.optimizers[0]
            elif data['model_name'] == 'stable_diffusion xl':
                pass
            elif data['model_name'] == 'stable_diffusion xl-turbo':
                optimizer = server.optimizers[2]
            elif data['model_name'] == 'stable_diffusion-turbo':
                pass
            else:
                raise ValueError('Invalid model name')
            
            img = data.get('img_in',None)
            if img is not None:
                img = Image.fromarray(np.uint8(img))
            req = sd_request(
                        state=optimizer.state,
                        prompt=data['prompt'],
                        video_task=False,
                        image = img,
                        num_samples=1,
                    )
            optimizer.waitlists[0].append(req)
            print('Request added')
            while req.output is None:
                await asyncio.sleep(0.1)
            img = Image.fromarray(np.uint8(req.output))
            image_stream = BytesIO()
            img.save(image_stream, format='JPEG')
            image_stream.seek(0)
            print('Request sending')
            return send_file(image_stream, mimetype='image/jpeg')
        
        elif data['model_name'] == 'Stable Video Diffusion':
            optimizer = server.optimizers[1]
            img = Image.fromarray(np.uint8(data['img_in']))
            req = sd_request(
                        state=optimizer.state,
                        prompt=data['prompt'],
                        video_task=True,
                        image=img,
                        num_samples=1,
                    )
            optimizer.waitlists[0].append(req)
            print('Request added')
            while req.output is None:
                await asyncio.sleep(0.1)
            return send_file(req.output, mimetype='video/mp4')
        
    elif request.method == 'GET':
        return 'Received GET request'

if __name__ == '__main__':
    server = sd_server(config_files=['configs/sd_21_512.yaml',
                                     'configs/svd.yaml',
                                     'configs/sdxl_turbo.yaml'
                                     ])
    backend_process = threading.Thread(target=server.runtime, daemon=True)
    backend_process.start()
    app.run()   # Start the server