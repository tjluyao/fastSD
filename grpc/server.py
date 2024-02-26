# We will create a gRPC server that will listen for requests and send responses to the client.
import asyncio
from concurrent import futures
import grpc
from PIL import Image
import numpy as np
from io import BytesIO
import conn_pb2, conn_pb2_grpc
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
from optimizer_sd import sd_optimizer, sd_request

class sd_server(sd_optimizer):
    def __init__(self, config_file):
        super().__init__(config_file)

    async def runtime(self):
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

class TestService(conn_pb2_grpc.Txt2ImgServiceServicer):
    def __init__(self):
        pass
    async def Txt2Img(self, request, context):
        prompt = request.prompt
        req = sd_request(
                        state=optimizer.state,
                        prompt=prompt,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
        optimizer.waitlists[0].append(req)
        while req.output is None:
            await asyncio.sleep(0.1)
        image_data = Image.fromarray(np.uint8(req.output))
        image_stream = BytesIO()
        image_data.save(image_stream, format='JPEG')
        return conn_pb2.ImageResponse(image_data=image_stream.getvalue())
    
async def service():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    conn_pb2_grpc.add_Txt2ImgServiceServicer_to_server(TestService(),server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("start service...")
    try:
        while True:
            asyncio.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        loop.close()

if __name__ == '__main__':
    optimizer = sd_server('configs/sd_21_512.yaml')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(service(),optimizer.runtime()))
    loop.close()