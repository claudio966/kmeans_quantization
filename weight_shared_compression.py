import scipy.io, os
import numpy as np
from scipy.io import savemat

from kneed import KneeLocator
from sklearn.cluster import KMeans

class Dnn_compressor:
    def __init__(self):
        self.frames = 300 

    def _get_clusters_numbers(self, pixels_data):
        kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 100,
        "random_state": 42,
        }

        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(pixels_data)
            sse.append(kmeans.inertia_)

        kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        
        return kl.elbow

    def _kmeans_model(self, pixels_data):
        n_clusters = self._get_clusters_numbers(pixels_data)
        if n_clusters is None:
            n_clusters = 1
        kmeans = KMeans(
            init="random",
            n_clusters=n_clusters,
            n_init=10,
            max_iter=100,
            random_state=42)
        
        kmeans.fit(pixels_data)
        
        return kmeans


    def enconder(self, example_idx:str, split_idx: str):        
        frame_idx_str = '000'
        frame_idx_int = 0

        output_folder = f'./compressed/data/example_{example_idx}/scores_mat/split_{split_idx}/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Go through all frames
        for frame in range(self.frames):
            print(f'Compressing frame {frame}')
            # load file
            frame_data = scipy.io.loadmat(f'./dataset/example_{example_idx}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
            frame_data = np.squeeze(frame_data['data'])
            shape = np.squeeze(frame_data.shape)

            frame_data_quantized = []
            # Train model
            kmeans = self._kmeans_model(frame_data[0].flatten().reshape(-1, 1))
            frame_data_quantized.append(kmeans.labels_)

            for i in range(1, shape[0]):
                pixels = frame_data[i]
                pixels = pixels.flatten()
                pixels = pixels.reshape(-1,1)
                labels = kmeans.predict(pixels)
                frame_data_quantized.append(labels)

            # enconder quantized channels
            frame_data_compressed = np.array(frame_data_quantized, np.int8)
            

            # save file
            data_dic = {"data": frame_data_compressed, "centers": kmeans.cluster_centers_, "orig_shape": shape}
            savemat(f'{output_folder}/scores_frame_{frame_idx_str}.mat', data_dic)
            
            # update frame count
            frame_idx_int += 1
            if frame_idx_int < 10:
                frame_idx_str = '00' + str(frame_idx_int)
            elif frame_idx_int < 100:
                frame_idx_str = '0' + str(frame_idx_int)
            else:
                frame_idx_str = str(frame_idx_int)

    def decoder(self, example_idx: str, split_idx: str):
        frame_idx_str = '000'
        frame_idx_int = 0

        output_folder = f'./uncompressed/example_{example_idx}/scores_mat/split_{split_idx}/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Go through all frames
        for frame in range(self.frames):
            print(f'Decompressing frame {frame}')
            # load file
            frame_data = scipy.io.loadmat(f'compressed/data/example_{example_idx}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
            frame_data_compressed = np.array(frame_data['data'], np.float32)
            # Retrieve the original shape for reconstruction
            frame_data_orig_shape = np.squeeze(frame_data['orig_shape'])
            frame_data_decoder = np.squeeze(frame_data['centers'])
            #frame_data_original_shape = frame_data['shape']
            shape = frame_data_compressed.shape

            frame_data_uncompressed = []
            for i in range(shape[0]):
                for decoder in range(len(frame_data_decoder)):
                    frame_data_compressed[i][frame_data_compressed[i] == decoder] = frame_data_decoder[decoder]
                frame_data_uncompressed.append(frame_data_compressed[i].reshape((frame_data_orig_shape[1], frame_data_orig_shape[2])))
            frame_data_uncompressed = np.array(frame_data_uncompressed, np.float32)
            
            # save file
            data_dic = {"data": frame_data_uncompressed, "label":frame_idx_str}
            savemat(f'{output_folder}/scores_frame_{frame_idx_str}.mat', data_dic)
            
            # update frame count
            frame_idx_int += 1
            if frame_idx_int < 10:
                frame_idx_str = '00' + str(frame_idx_int)
            elif frame_idx_int < 100:
                frame_idx_str = '0' + str(frame_idx_int)
            else:
                frame_idx_str = str(frame_idx_int)