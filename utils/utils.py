import os.path
from os import listdir
from os.path import isfile, join
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PrepareDataset:
    def __init__(self, img_path, test_path, save_path, block_size, size = (256, 256)) -> None:
        self.img_path = img_path
        self.save_path = save_path
        self.size = size
        self.block_size = block_size
        self.img_fnames = []
        directories = next(os.walk(img_path), (None, None, []))[1]
        for directory in tqdm(directories):
            self.img_fnames.extend(next(os.walk(img_path + directory), (None, None, []))[2])
        self.test_path = test_path
        self.img_test_fnames = next(os.walk(test_path), (None, None, []))[2]
        

    def create_dataset_autoencoder(self):
        for img_names in self.img_fnames:
            dir_name = img_names[:9]
            image = self._preprocess_image(join(self.img_path + dir_name, img_names))
            missing_pixel_image = self._create_missing_pixel_img(image)
            # imshow(missing_pixel_image, cmap='gray')
            # imshow(missing_pixel_image[:8, :8], cmap='gray', dpi = 5)
            # print(missing_pixel_image[:8, :8])
            cv.imwrite(self.save_path + img_names[:-3] + '.jpg', missing_pixel_image, [cv.IMWRITE_JPEG_QUALITY, 100])
            # break

    def parse_image_2d(self, img):
        pass

    def create_test_original(self):
        i=0
        X = np.zeros((1024 * len(self.img_test_fnames), 8, 8))
        for img_names in tqdm(self.img_test_fnames):
            image = self._preprocess_image(join(self.test_path, img_names))
            tiled_image = self._reshape_split(image)
            X[i*1024:(i+1)*1024, :, :] = tiled_image
            i += 1
        y = X.copy()[:, 3:5, 3:5].reshape(1024 * len(self.img_test_fnames), -1)
        X[:, 3:5, 3:5] = np.array([[None, None],
                                   [None, None]])
        return X[~np.isnan(X)].reshape(-1, 60) / 255., y / 255.

    def create_test_2d(self):
        i=0
        X = np.zeros((1024 * len(self.img_test_fnames), 8, 8))
        for img_names in tqdm(self.img_test_fnames):
            image = self._preprocess_image(join(self.test_path, img_names))
            tiled_image = self._reshape_split(image)
            X[i*1024:(i+1)*1024, :, :] = tiled_image
            i += 1
        y = X.copy()[:, 3:5, 3:5].reshape(1024 * len(self.img_test_fnames), -1)
        X[:, 3:5, 3:5] = np.array([[0., 0.],
                                   [0., 0.]])
        return X / 255., y / 255.
    
    def test_model_pytorch(self, model):
        image = self._preprocess_image('./dataset/original_image/balloon.bmp')
        tiled_image = self._reshape_split(image)
        X = tiled_image.copy()
        X[:, 3:5, 3:5] = np.array([[None, None],
                                   [None, None]])
        test = X[~np.isnan(X)].reshape(-1, 60) / 255.
        prediction_result = model.predict(test).reshape(-1, 2, 2) * 255.
        X[:, 3:5, 3:5] = prediction_result
        return X.reshape(32, 32, 8, 8).swapaxes(1, 2).reshape(256, 256)

    def create_dataset_original(self, sample_size):
        i=0
        X = np.zeros((1024 * sample_size, 8, 8))
        for img_names in tqdm(self.img_fnames[:sample_size]):
            dir_name = img_names[:9]
            image = self._preprocess_image(join(self.img_path + dir_name, img_names))
            tiled_image = self._reshape_split(image)
            X[i*1024:(i+1)*1024, :, :] = tiled_image
            i += 1
        y = X.copy()[:, 3:5, 3:5].reshape(1024 * sample_size, -1)
        X[:, 3:5, 3:5] = np.array([[None, None],
                                   [None, None]])
        return X[~np.isnan(X)].reshape(-1, 60) / 255., y / 255.
    
    def create_dataset_2d(self, sample_size):
        i=0
        X = np.zeros((1024 * sample_size, 8, 8))
        for img_names in tqdm(self.img_fnames[:sample_size]):
            dir_name = img_names[:9]
            image = self._preprocess_image(join(self.img_path + dir_name, img_names))
            tiled_image = self._reshape_split(image)
            X[i*1024:(i+1)*1024, :, :] = tiled_image
            i += 1
        y = X.copy()[:, 3:5, 3:5].reshape(1024 * sample_size, -1)
        X[:, 3:5, 3:5] = np.array([[0., 0.],
                                   [0., 0.]])
        return X / 255., y / 255.
    
    def create_dataset_ircnn(self, sample_size):
        i=0
        X = np.zeros((64 * sample_size, 32, 32))
        y = np.zeros((64 * sample_size, 32, 32))
        for img_names in tqdm(self.img_fnames[:sample_size]):
            dir_name = img_names[:9]
            image = self._preprocess_image(join(self.img_path + dir_name, img_names))
            missing_pixel_image = self._create_missing_pixel_img(image)
            # tiled_image = self._reshape_split(image)
            X[i*64:(i+1)*64, :, :] = self._reshape_split(missing_pixel_image)
            y[i*64:(i+1)*64, :, :] = self._reshape_split(image)
            i += 1
        # y = X.copy()
        # X[:, 3:5, 3:5] = np.array([[0., 0.],
        #                            [0., 0.]])
        return X / 255., y / 255.
    
    def _reshape_split(self, img):
        if img.ndim == 2:
            img_height, img_width = img.shape
            channels = 1
        if img.ndim == 3:
            img_height, img_width, channels = img.shape
        tile_height, tile_width = self.block_size, self.block_size

        tiled_array = img.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width)
        
        tiled_array = tiled_array.swapaxes(1, 2)
        return tiled_array.reshape(-1, tile_height, tile_width)
    
    def _preprocess_image(self, img_full_path):
        img = cv.imread(img_full_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, self.size, cv.INTER_LINEAR)
        img = img.astype(np.float64)
        return img

    def _create_missing_pixel_img(self, img):
        temp_img = img.copy()
        for r in range(self.block_size - 4 - 1, img.shape[0], self.block_size):
            for c in range(self.block_size - 4 - 1, img.shape[1], self.block_size):
                temp_img[r, c], temp_img[r + 1, c], temp_img[r, c + 1], temp_img[r + 1, c + 1] = 0, 0, 0, 0
        return temp_img



def imshow(img, cmap=None, vmin=0, vmax=255, frameon=False, dpi=72):
    fig = plt.figure(figsize=[img.shape[1]/dpi, img.shape[0]/dpi], \
                    frameon=frameon)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()

def compare_img(img1, img2):
    pass


# def is_missing_pixel(r, c):
#     return (r >= M1 and c >= M1 and r <= M2 and c <= M2)

# def do_parse_img(img, inps, tgts):
#     num_rows = img.shape[0]
#     num_cols = img.shape[1]

#     inp = np.zeros((NUM_INP_NODES))
#     tgt = np.zeros((NUM_OUT_NODES))

#     for r0 in range(0, num_rows, BLOCK_SIZE):
#         for c0 in range(0, num_cols, BLOCK_SIZE):
#             # extract and reshape a block of the image
#             pos1 = 0
#             pos2 = 0
#             for r in range(BLOCK_SIZE):
#                 for c in range(BLOCK_SIZE):
#                     assert(r0 + r < num_rows)
#                     assert(c0 + c < num_cols)

#                     if is_missing_pixel(r, c):
#                         tgt[pos2] = img[r0 + r, c0 + c]
#                         pos2 += 1
#                     else:
#                         inp[pos1] = img[r0 + r, c0 + c]
#                         pos1 += 1

#                 # add the input and target patterns
#                 inps.append(inp.copy() / 255.0)
#                 tgts.append(tgt.copy() / 255.0)


