import numpy as np
from matplotlib import pyplot as plt
import scipy
import glob

def get_mse(target, predicted):
    MSE = np.square(np.subtract(target, predicted)).mean()
    
    return MSE

def plot_hist(example_idx: str, split_idx: str):
    frame_idx_str = '000'
    all_channel = True

    original_data = scipy.io.loadmat(f'dataset/example_{example_idx}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
    original_data = np.squeeze(original_data['data'])

    if all_channel:
        original_data = original_data.flatten()
        plt.title(f'Split {split_idx}: all channel histogram')
        plt.ylabel('Occurrences')
        plt.xlabel('Weight')
        plt.hist(original_data)
        plt.savefig(f'histograms/split_{split_idx}_all_channel_hist')
        plt.close()
    else:
        n_channels = 5
        for channel in range(n_channels):
            plt.title(f'Split {split_idx}: channel {channel} histogram')
            plt.xlabel('Occurrences')
            plt.ylabel('Weight')
            plt.hist(original_data[channel].flatten())
            plt.savefig(f'histograms/split_{split_idx}_hist_{channel}')
            plt.close()


def plot_mse(frames, split_idx: str):
    # Go through all examples
    for example in range(1, len(glob.glob('./dataset/example*'))):
        frame_idx_str = '000'
        frame_idx_int = 0
        print(f'[INFO] Plotting example {example} MSE')
        # Go through all frames
        mses = np.array([])
        for _ in range(frames):
            # load file
            original_data = scipy.io.loadmat(f'dataset/example_0{example}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
            original_data = np.squeeze(original_data['data'])
            
            uncompressed_data = scipy.io.loadmat(f'uncompressed/example_0{example}/scores_mat/split_{split_idx}/scores_frame_{frame_idx_str}.mat')
            uncompressed_data = np.squeeze(uncompressed_data['data'])

            mses = np.append(mses, get_mse(original_data, uncompressed_data))        
            # update frame count
            frame_idx_int += 1
            if frame_idx_int < 10:
                frame_idx_str = '00' + str(frame_idx_int)
            elif frame_idx_int < 100:
                frame_idx_str = '0' + str(frame_idx_int)
            else:
                frame_idx_str = str(frame_idx_int)

        print(max(mses))
        with open('mses_results.txt', 'a') as f:
            f.write(str(max(mses)) + "\n")

        plt.title(f'MSE of Example: 0{example} - Split: {split_idx}')
        plt.xlabel('Frames')
        plt.ylabel('MSE')
        plt.plot(mses, label=f'Example 0{example}')
        plt.savefig(f'results/Example_{example}_MSE.png')
        plt.close() 
