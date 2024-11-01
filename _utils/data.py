import scipy.io as sio
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from tqdm import tqdm

# loads stim file as an array of 3x224x224 images
def load_stim_file(path, stim_var='stim', model='vgg16'):
    mirror_data = sio.loadmat(path, squeeze_me=True, struct_as_record=True)
    stim_data = []
    print("Loading stimuli ...")
    for i, stim in enumerate(tqdm(mirror_data[stim_var])):
        h,w = stim.shape[0],stim.shape[1]
        # if 1 channel, convert to 3 channels
        if len(stim.shape) == 2:
            # repeat the image 3 times
            stim = np.repeat(stim[:, :, np.newaxis], 3, axis=2)
        # print(stim)
        stim = stim.astype(np.uint8)
        img = Image.fromarray(stim)
        padding_transform = None
        
        if h > w: 
            padding_transform = transforms.Pad(((h-w)// 2, 0 ))  # Padding to make img square (only pad width)
        elif w > h:
            padding_transform = transforms.Pad((0 , (w-h) // 2))  # Padding to make img square (only pad height)
        else:
            padding_transform = transforms.Pad((0, 0))

        img_size = 224
        if 'face_vit' in model:
            img_size = 112
            
        transform = transforms.Compose([
            padding_transform,
            transforms.Resize((img_size, img_size)),  # Resize to 224x224
            transforms.ToTensor()
        ])

        img = transform(img)

        # img = img.numpy() * 255
        # img = img.astype(np.uint8
        # img_pil = np.transpose(img, (1, 2, 0))
        # print(img_pil.shape)
        # img_pil = Image.fromarray(img_pil)
        # # display image
        # plt.imshow(img_pil)
        # plt.show()    

        stim_data.append(img)
        
    stim_data = np.array(stim_data)
    return stim_data

def display_np_image(img_arr):
    plt.imshow(torch.from_numpy(img_arr).permute(1,2,0))
    plt.show()

def load_multiple_stim_files(path):
    try :
        stim_data = sio.loadmat(path, squeeze_me=True, struct_as_record=True)
    except:
        file = h5py.File(path, 'r')
        # print(file.keys())
        stim_data = {}
        for i, key in enumerate(file.keys()):
            if i > 0:
                ref = file[key]
                stim_data[key] = np.array(ref)
                    
    return stim_data

def save_stims(stims, path='./images'):
    num_stim = stims.shape[0]
    for stim_i in range(num_stim):
        # save image
        img = stims[stim_i]
        img = np.transpose(img, (1, 2, 0))
        plt.imsave(f'{path}/{stim_i}.png', img)