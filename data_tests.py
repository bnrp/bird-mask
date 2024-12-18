dataset_path = 'nabirds-data/nabirds/'
image_path = dataset_path + 'images/'

import os, os.path
import numpy as np

folder_list = os.listdir(image_path)
file_list = []

test_frac = 0.2

test = 0
train = 0

open('train_test_split_good.txt', 'w').close()

with open('train_test_split_good.txt', 'a') as f:
    for folder in folder_list:
        file_list = os.listdir(image_path+folder+'/')
        length = int(np.round(test_frac*int(len(file_list))))
    
        file_list_train = file_list[length:]
        file_list_test = file_list[0:length]

        for file in file_list_train:
            train += 1
            id = file[:-4]
            id = id[0:8] + '-' + id[8:12] + '-' + id[12:16] + '-' + id[16:20] + '-' + id[20:]

            f.write(id + ' 1\n')

        for file in file_list_test:
            test += 1
            id = file[:-4]
            id = id[0:8] + '-' + id[8:12] + '-' + id[12:16] + '-' + id[16:20] + '-' + id[20:]

            f.write(id + ' 0\n')
        
f.close()        

print('Train/Test split: ' + str(100-100*test_frac) + '/' + str(100*test_frac))
print('Training samples: ' + str(train))
print('Testing samples: ' + str(test))
