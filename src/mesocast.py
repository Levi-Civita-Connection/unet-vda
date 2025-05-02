import sys
import numpy as np
import tensorflow as tf
import os
import lz4.frame
import pdb
from concurrent import futures
import grpc
import nr2_types_pb2
import nr2_types_pb2_grpc
import unittest
from dealias import VelocityDealiaser
from feature_extraction import create_downsampler, create_upsampler

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.ticker as ticker
from plotting import create_polar_plot, add_cbar

sys.path.append('src/')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "True"


start_neurons_az = 16
vel = tf.keras.Input(shape=(None, None, 1))

down = create_downsampler(inp=vel, start_neurons=start_neurons_az,
                          input_channels=1)
up = create_upsampler(n_inputs=1, start_neurons=start_neurons_az, n_outputs=6)

vda = VelocityDealiaser(down, up)

vda.load_weights('models/dealias_sn16_csi9764.SavedModel')


class MesoDealiaser(nr2_types_pb2_grpc.MesoDealiaser):
    def DealiasRadarFrame(self, request, context):
        nyq = request.NyquistVelocity
        l2 = request.Gates
        n_frames = 1
        reshaped_l2 = normalize_data(l2, request.RadialCount,
                                     request.GateCount)
        l2 = reshaped_l2.copy()
        nyq = np.array([[nyq]])

        if l2.shape[2] == 720:
            s = l2.shape
            l2 = np.reshape(l2, (s[0], s[1], s[2]//2, 2, s[3], s[4]))
            l2 = np.transpose(l2, (0, 3, 1, 2, 4, 5))
            l2 = np.reshape(l2, (2*s[0], s[1], s[2]//2, s[3], s[4]))
            nyq = np.stack((nyq, nyq))
            nyq = np.transpose(nyq, (1, 0, 2))
            nyq = np.reshape(nyq, (-1, n_frames, 1))
            recombine = True
        else:
            recombine = False

        pad_deg = 12
        l2 = np.concatenate((l2[:, :, -pad_deg:, :, :], l2, l2[:, :, :pad_deg,
                             :, :]), axis=2)
        l2[(l2 <= -64) | (l2 == 131072) | (l2 == 131071)] = np.nan

        l2 = l2[:, :, :, :1152, :]
        inp = {'vel': l2, 'nyq': nyq}
        out = vda.predict(inp)
        dealiased_vel = out['dealiased_vel'].copy()

        if recombine:
            s = dealiased_vel.shape
            dealiased_vel = np.reshape(dealiased_vel, (s[0]//2, 2, s[1], s[2],
                                                       s[3]))
            dealiased_vel = np.transpose(dealiased_vel, (0, 2, 1, 3, 4))
            dealiased_vel = np.reshape(dealiased_vel, (s[0] // 2, -1, s[2],
                                                       s[3]))
        unnormalized_data = unnormalize_data(dealiased_vel)

        response = nr2_types_pb2.RadarFrame()
        response.CopyFrom(request)

        clean_data = np.nan_to_num(unnormalized_data, nan=131071)
        reshaped = clean_data.reshape((752, 1152))
        flipped = np.flip(reshaped, axis=0)
        response.Gates.extend(flipped.flatten().tolist())
        response.RadialCount = dealiased_vel.shape[1]
        response.GateCount = dealiased_vel.shape[2]
        # Set Colormap
        # cmap=cm.get_cmap('seismic').copy()
        # cmap.set_under([.9,.9,.9])
        # cmap.set_bad([.9,.9,.9])
        # cmap.set_over([.9,.9,.9])
        # norm=mpl.colors.Normalize(vmin=-70, vmax=70)  


        # fig,axs=plt.subplots(1,3,figsize=(15,5))
        # im,_ = create_polar_plot(axs[0], out['alias_mask'][0,:,:,0],cmap,norm,0.5,annotate=False)
        # im,_ = create_polar_plot(axs[1], out['dealiased_vel'][0,:,:,0],cmap,norm,0.5,annotate=False)
        # add_cbar(fig,axs[2],im,'m/s')

        # for k,ax in enumerate(axs):
        #         ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        #         ax.xaxis.set_major_formatter(ticks_x)
        #         ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        #         ax.yaxis.set_major_formatter(ticks_y)
        #         ax.set_xlabel('km')
        # axs[0].set_ylabel('km')
        # axs[0].set_title('Level 2 Velocity (Aliased)')
        # axs[1].set_title('U-Net Result')
        # axs[2].set_title('Level 3 Velocity Reference')

        # plt.tight_layout()
        # plt.savefig('dealiasing.png', dpi=300)

        return response
        

class TestMesoDealiaser(unittest.TestCase):
    def test_normalize_data(self):
        data = np.random.rand(720 * 1192).astype(np.float32)
        reshaped_data = normalize_data(data, 720, 1192)
        self.assertEqual(reshaped_data.shape, (1, 1, 720, 1192, 1))
        self.assertEqual(reshaped_data[0, 0, 0, 500, 0], data[500])
        self.assertEqual(reshaped_data[0, 0, 0, 0, 0], data[0])
        self.assertEqual(reshaped_data[0, 0, 1, 0, 0], data[1192])
        self.assertEqual(reshaped_data[0, 0, 2, 53, 0], data[1192*2+53])
        self.assertEqual(reshaped_data[0, 0, -1, -1, 0], data[-1])

    def test_dealias_radar_frame(self):
        with open("1746168734711", 'rb') as f:
            compressed_data = f.read()

        compressed_data = compressed_data[4:]

        decompressed_data = lz4.frame.decompress(compressed_data)

        request = nr2_types_pb2.RadarFrame()
        request.ParseFromString(decompressed_data)

        meso_dealiaser = MesoDealiaser()
        response = meso_dealiaser.DealiasRadarFrame(request, None)
        self.assertIsInstance(response, nr2_types_pb2.RadarFrame)


def normalize_data(data, radials, gates):
    normalized_data = np.zeros((1, 1, radials, gates, 1), dtype=np.float32)
    for i in range(len(data)):
        normalized_data[0, 0, i // gates, i % gates, 0] = data[i]
    return normalized_data


def unnormalize_data(data):
    return data.reshape(-1)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nr2_types_pb2_grpc.add_MesoDealiaserServicer_to_server(MesoDealiaser(),
                                                           server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
