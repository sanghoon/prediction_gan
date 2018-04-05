import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# Originally from https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/celebA_data_preprocess.py
# Added naive center crop

# root path depends on your computer
root = 'img_align_celeba/'
save_root = 'celeba_align_resized/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])

    w, h, c = img.shape
    cropsize = int(min(w, h) * 0.9) // 2

    # Center crop
    img = img[(w//2 - cropsize):(w//2 + cropsize), (h//2 - cropsize):(h//2 + cropsize), :]
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)