import grpc
import nr2_types_pb2
import nr2_types_pb2_grpc
import lz4.frame

options = [
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
    ('grpc.max_send_message_length', 100 * 1024 * 1024)      # Also increase if your requests are large
]


def run():
    with grpc.insecure_channel('localhost:50051', options=options) as channel:
        stub = nr2_types_pb2_grpc.MesoDealiaserStub(channel)

        with open("1746168734711", 'rb') as f:
            compressed_data = f.read()

        compressed_data = compressed_data[4:]

        decompressed_data = lz4.frame.decompress(compressed_data)

        request = nr2_types_pb2.RadarFrame()
        request.ParseFromString(decompressed_data)
        print("nyquist velocity: ", request.NyquistVelocity)

        response = stub.DealiasRadarFrame(request)
    print(f"client received: {response.NyquistVelocity}")


if __name__ == '__main__':
    run()
