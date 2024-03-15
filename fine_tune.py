#%%
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm

ch_org = True
ch_diff = True
ch_bg = True


@tf.function()
def random_jitter( X, y, train, transfer=False):            
    # # Random cropping back to 128x128
    # X = tf.image.random_crop(X, size=x_shape, seed=42)
    # y = tf.image.random_crop(y, size=y_shape, seed=42)

    # Random mirroring
    X = tf.image.random_flip_left_right(X, seed=43)
    y = tf.image.random_flip_left_right(y, seed=43)
    
    # Random mirroring
    X = tf.image.random_flip_up_down(X, seed=44)
    y = tf.image.random_flip_up_down(y, seed=44)
    
    if transfer:
        #random invert first two colors
        if tf.random.uniform(shape=(), minval=0, maxval=1) > 0.5:
            X = X * [-1, -1, 1] + [1, 1, 0] 
                       
    if train:
        X = tf.image.random_brightness(X, max_delta=0.15, seed=45)
        X = tf.image.random_contrast(X, lower=0.85, upper=1.15, seed=47)
        
    k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    X = tf.image.rot90(X, k=k*2)
    y = tf.image.rot90(y, k=k*2)
    return X, y


def get_random_index( A, B):
    a, b = np.random.randint(0, len(A)), np.random.randint(0, len(B))
    if a == b:
        if b == 0:
            b = 1
        elif b == len(B)-1:
            b = len(B)-2
        else:
            b = np.random.choice([b-1, b+1])  
    return a, b

    
def _generator(examples, backgrounds, labels, ch_org=True, ch_diff=True, ch_bg=True):
    i = np.random.randint(0, len(examples))
    _X = []
    
    if np.random.uniform(0,1) > 0.8:
        a, b = get_random_index(backgrounds[i], backgrounds[i])
        example = backgrounds[i][a].copy()
        label = np.zeros_like(labels[i])    
    else:
        a, b = get_random_index(examples[i], backgrounds[i])
        example = examples[i][a].copy()
        label = labels[i].copy()
        
    if ch_org:
        _X.append(example)
    if ch_diff:
        _diff = example - backgrounds[i][b].copy()
        _diff = (_diff - _diff.min()) / (_diff.max() - _diff.min())
        _X.append(_diff)                         
    if ch_bg:
        _X.append(backgrounds[i][b].copy())
    
    return np.dstack(_X), label



# zero pad to 256, 512
def pad(img):
    if img.shape[0] < 256:
        img = np.pad(img, ((0,256-img.shape[0]),(0,512-img.shape[1])), 'constant', constant_values=0)
    else:
        img = np.pad(img, ((0,0),(0,512-img.shape[1])), 'constant', constant_values=0)
    return img

def _load_data(i):
    
    example_path, label_path, background_path = f"data/highspeed_train/example{i}", f"data/highspeed_train/label{i}", f"data/highspeed_train/background{i}"
    label = cv2.imread(label_path + '.png', cv2.IMREAD_UNCHANGED)
    label = np.max(label, axis=2).astype(np.float32) / 255
    label = pad((label>0.5).astype(np.float32))
        
    examples, backgrounds = [], []
    for x in os.listdir(example_path):
        examples.append(pad(cv2.imread(example_path + f'/{x}', cv2.IMREAD_GRAYSCALE) / 255))
    
    for x in os.listdir(background_path):
        backgrounds.append(pad(cv2.imread(background_path + f'/{x}', cv2.IMREAD_GRAYSCALE) / 255))

    return examples, label[:,:,np.newaxis], backgrounds

def _load_val(i):    
    path = f"data/highspeed_train/validation{i}"
    example = pad(cv2.imread(f'{path}/example{i}.tiff', cv2.IMREAD_GRAYSCALE) / 255)
    label = cv2.imread(f"{path}/label{i}.png", cv2.IMREAD_UNCHANGED)
    label = np.max(label, axis=2).astype(np.float32) / 255
    label = pad((label>0.5).astype(np.float32))
    return [example], label[:,:,np.newaxis]

def _load_test(path):
    X, Y, BG = [], [], []
    for file in sorted(os.listdir(f'{path}/X'), key=lambda x: int(x.split("_")[-1].split(".")[0])):
        img = cv2.imread(f'{path}/X/{file}', cv2.IMREAD_GRAYSCALE)
        backgrounds = os.listdir(f'{path}/background')
        background = cv2.imread(f'{path}/background/{backgrounds[np.random.randint(len(backgrounds))]}', cv2.IMREAD_GRAYSCALE)
        
        # split file name to get number
        file = int((f'{path}/X/{file}').split('.')[0].split('_')[-1])
        lab = 1 - cv2.imread(f'{path}/Y/{file}.jpg', cv2.IMREAD_GRAYSCALE)/255
        lab = increaseWidth(lab)
        lab = (lab > 0.5).astype(np.float32)

        BG.append(pad(background)/255)
        X.append(pad(img)/255)
        Y.append(pad(lab)[:,:,np.newaxis])
    return X, Y, BG


def _load_transfer(path, ch_org=True, ch_diff=True, ch_bg=True):
    X, Y = [], []
    X_val, Y_val = [], []
    
    # iterate subfolders
    for folder in tqdm(os.listdir(path)):
        # get amount of images in folder
        num_images = np.sqrt(len(os.listdir(f'{path}/{folder}'))/3)

        for i in range(int(num_images)):
            for j in range(int(num_images)):
                example = pad(cv2.imread(f'{path}/{folder}/img_{i}_{j}.jpg', cv2.IMREAD_GRAYSCALE) / 255)
                background = pad(cv2.imread(f'{path}/{folder}/orig_{i}_{j}.jpg', cv2.IMREAD_GRAYSCALE) / 255)                
                lab = cv2.imread(f'{path}/{folder}/lab_{i}_{j}.jpg', cv2.IMREAD_GRAYSCALE) / 255
                lab = pad(increaseWidth(lab, width=5))[:,:,np.newaxis]
                lab = (lab>0.5).astype(np.float32)
                
                _X = test_generator(example, background, ch_org=ch_org, ch_diff=ch_diff, ch_bg=ch_bg)
                
                if np.random.rand() < 0.7:
                    X.extend([_X[:256].copy(), _X[-256:].copy()])
                    Y.extend([lab[:256].copy(), lab[-256:].copy()])
                else:
                    X_val.extend([_X[:256].copy(), _X[-256:].copy()])
                    Y_val.extend([lab[:256].copy(), lab[-256:].copy()])

    return X, Y, X_val, Y_val
    
    
def test_generator(example, background, ch_org=True, ch_diff=True, ch_bg=True):
    _X = []
        
    if ch_org:
        _X.append(example.copy())
    if ch_diff:
        _diff = example.copy() - background.copy()
        _diff = (_diff - _diff.min()) / (_diff.max() - _diff.min())
        _X.append(_diff)                         
    if ch_bg:
        _X.append(background.copy())
    
    return np.dstack(_X)

#%%

#increase width of lines in labels
def increaseWidth(img, width=3):
    kernel = np.ones((width,width), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img