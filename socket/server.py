import socket
from optimizer_sd import sd_optimizer, sd_request
import threading, time
from PIL import Image
import numpy as np
from io import BytesIO

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

class conn_thread(threading.Thread):
    def run(self):
        hostname = '127.0.0.1'          
        port = 6666                     
        addr = (hostname,port)
        srv = socket.socket()           
        srv.bind(addr)
        srv.listen(5)
        print("waiting connect")
        while True:
            connect_socket,client_addr = srv.accept()
            print(client_addr)
            recevent = connect_socket.recv(1024)
            text = str(recevent,encoding='gbk')
            print(text)
            req = sd_request(
                        state=optimizer.state,
                        prompt=text,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
            optimizer.waitlists[0].append(req)
            while req.output is None:
                time.sleep(0.1)
            
            img = Image.fromarray(np.uint8(req.output))
            image_stream = BytesIO()
            img.save(image_stream, format='JPEG')
            image_data = image_stream.getvalue()

            image_size = len(image_data)
            connect_socket.send(image_size.to_bytes(4, 'big'))

            connect_socket.send(image_data)
            connect_socket.close()

if __name__ == '__main__':
    optimizer = sd_server('configs/sd_21_512.yaml')
    backend_thread = conn_thread()
    backend_thread.start()
    while True:
        optimizer.runtime()