import csv
import numpy as np
import os
import cv2 as cv
from keras.utils import Sequence,to_categorical

class AFLWFaceRegionsSequence(Sequence):
    def __init__(self, batch_size, regions_csv_file_name, path_to_image_folder, image_size, rotate=None):
        self.regions_csv_file_name = regions_csv_file_name
        self.batch_size = batch_size
        self.path_to_image_folder = path_to_image_folder
        self.image_size = image_size
        self.rotate = rotate

    def __len__(self):
        with open(self.regions_csv_file_name) as f:
            i = 0
            for i,l in enumerate(f):
                pass
            return int(np.ceil(i / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        with open(self.regions_csv_file_name) as f:
            line_num = 0
            prev_image_file_name = ''
            image = None

            for row in csv.reader(f):
                image_file_name = row[0]
                x = int(row[1])
                y = int(row[2])
                width = int(row[3])
                height = int(row[4])
                is_face = row[5] == 'True'

                if line_num >= idx*self.batch_size:
                    if image_file_name != prev_image_file_name:
                        image = cv.imread(os.path.join(self.path_to_image_folder,image_file_name))
                        prev_image_file_name = image_file_name

                    regional_image = cv.resize(image[y:y+height,x:x+width],self.image_size)
                    if self.rotate is not None:
                        regional_image = cv.rotate(regional_image, self.rotate)
          
                    batch_x.append(regional_image)
                    batch_y.append(1 if is_face else 0)

                line_num += 1

                if line_num >= (idx+1) * self.batch_size:
                    break

        return np.array(batch_x),to_categorical(np.array(batch_y), num_classes=2,dtype='float32')
