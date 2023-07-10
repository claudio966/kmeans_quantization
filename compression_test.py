from weight_shared_compression import Dnn_compressor
import glob
from utils import plot_mse

compressor = Dnn_compressor()
compression = False
decompression = False
generate_plots = True

frames = 300
splits = ['01', '16']


# Start the compression state
if compression:
    # Compressing/Decompressing split 01
    print(' == compression Stage == ')
    for example in range(1, len(glob.glob('./dataset/example*'))):
        print(f'[INFO] Working on example {example}')
        example_idx = '0' + str(example)
        print(f'[INFO] Split 01')
        compressor.enconder(example_idx, splits[0])
        print(f'[INFO] Split 16')
        compressor.enconder(example_idx, splits[1])

# Start the decompression stage
if decompression:
    # Compressing/Decompressing split 16
    print(' == Decompression Stage == ')
    for example in range(6, len(glob.glob('./dataset/example*')) + 1):
        print(f'[INFO] Working on example {example}')
        example_idx = '0' + str(example)
        print(f'[INFO] Split 01')
        compressor.decoder(example_idx, splits[0])
        print(f'[INFO] Split 16')
        compressor.decoder(example_idx, splits[1])

# Generate mse plots
if generate_plots:
    print(f'Working on Split {splits[0]}')
    plot_mse(frames, splits[0])