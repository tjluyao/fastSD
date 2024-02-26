import conn_pb2, conn_pb2_grpc
import grpc

def run():
    conn=grpc.insecure_channel('localhost:50052')
    client = conn_pb2_grpc.Txt2ImgServiceStub(channel=conn)
    request = conn_pb2.ImageRequest(prompt="a cat")
    respnse = client.Txt2Img(request)
    print("received:",respnse.image_data)
if __name__ == '__main__':
    run()