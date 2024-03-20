import glob
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image 
import numpy as np


# train data loader
def train_dataloader(opt):
    # get data
    if opt.dataset == 'npz':
        noise_data,clear_data = load_npz_data(opt,'train')
        noise_val,clear_val = load_npz_data(opt,'val')
    else:
        noise_data,clear_data,noise_val,clear_val = load_data(opt)
    # create datasets 
    train_dataset = CustomDataset(noise_data,clear_data)
    val_dataset = CustomDataset(noise_val,clear_val)
    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader

# test data loader 
def test_dataloader(opt):
    # get data
    if opt.dataset == 'npz':
        noise_data,clear_data = load_npz_data(opt,'test')
    else:
        noise_data = load_test_data(opt)
        clear_data = noise_data
    # create datasets 
    test_dataset = CustomDataset(noise_data,clear_data)
    # create dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def load_npz_data(opt,data_mode):
    if data_mode == 'train':
        noise_dataset = np.load(opt.clean_dataroot + '1018paircropnoise.npz')
        clear_dataset = np.load(opt.clean_dataroot + '1018paircropclear.npz')
        
    elif data_mode == 'val':
        noise_dataset = np.load(opt.clean_dataroot + '1018paircropnoiseval.npz')
        clear_dataset = np.load(opt.clean_dataroot + '1018paircropclearval.npz')
    
    elif data_mode == 'test':
        noise_dataset = np.load(opt.clean_dataroot + '1018paircropnoisetest.npz')
        clear_dataset = np.load(opt.clean_dataroot + '1018paircropcleartest.npz')
    
    noise_data = noise_dataset['x_train']
    noise_data = np.array(noise_data).astype('float16')
    noise_data = noise_data.reshape([noise_data.shape[0],1,36,36]) / 255.0
    
    clear_data = clear_dataset['y_train']
    clear_data = np.array(clear_data).astype('float16')
    clear_data = clear_data.reshape([clear_data.shape[0],1,36,36]) / 255.0

    return noise_data,clear_data

def load_data(opt):

    clear_paths = []
    noise_paths = []

    clear_dataset = []
    noise_dataset = []

    clear_paths = _image_paths_search(opt.clean_dataroot)
    clear_paths.sort()
    noise_paths = _image_paths_search(opt.noise_dataroot)
    noise_paths.sort()

    for i in range(min(opt.max_dataset_size, len(clear_paths))):
        path = clear_paths[i]
        image = Image.open(path).convert("L")
        image = np.array(image)
        
        # apply gaussian blur
        # noise_image = cv2.GaussianBlur(image, (11,11), 0)
        clear_dataset.append(image)
        # noise_dataset.append(noise_image)

    for i in range(min(opt.max_dataset_size, len(noise_paths))):
        path = noise_paths[i]
        image = Image.open(path).convert("L")
        image = np.array(image)
        noise_dataset.append(image)

    noise_data = noise_dataset
    
    noise_data = np.array(noise_data).astype('float16')
    noise_data = noise_data.reshape([noise_data.shape[0],1,opt.input_h,opt.input_w]) / 255.0
    
    clear_data = clear_dataset
    clear_data = np.array(clear_data).astype('float16')
    clear_data = clear_data.reshape([clear_data.shape[0],1,opt.input_h,opt.input_w]) / 255.0

    # Splitting data into 1:9 ratio
    noise_data_val, noise_data_train = np.array_split(noise_data, [int(len(noise_data)*0.1)])
    clear_data_val, clear_data_train = np.array_split(clear_data, [int(len(clear_data)*0.1)])

    return noise_data_train, clear_data_train, noise_data_val, clear_data_val


def load_test_data(opt):

    noise_paths = []
    noise_dataset = []

    noise_paths = _image_paths_search(opt.noise_dataroot)
    noise_paths.sort()

    # load all noise image
    for i in range(min(opt.max_dataset_size, len(noise_paths))):
        path = noise_paths[i]
        image = Image.open(path).convert("L")
        image = np.array(image)
        noise_dataset.append(image)

    noise_data = noise_dataset
    
    noise_data = np.array(noise_data).astype('float16')
    noise_data = noise_data.reshape([noise_data.shape[0],1,opt.input_h,opt.input_w]) / 255.0

    return noise_data

def _image_paths_search(image_dirs):
        file_patterns = ['*.bmp', '*.png', '*.jpg', '*.jpeg']
        image_list = []
        print(image_dirs)
        for image_dir in image_dirs:
            temp_list = [file for pattern in file_patterns for file in glob.glob(os.path.join(image_dir, '**', pattern), recursive=True)]
            image_list.extend(temp_list)

        print(len(image_list))
        return image_list

class CustomDataset(Dataset):
    def __init__(self, noisy_data, clear_data):
        self.noisy_data = noisy_data
        self.clear_data = clear_data

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        noisy_image = self.noisy_data[idx]
        clear_image = self.clear_data[idx]
        return noisy_image, clear_image