from tensorflow import keras
import sys
sys.path.append('/home/francesco/projects/keras-preprocessing')
import keras_preprocessing.image
import numpy as np

# def reader_function(path):
#     img_array = pydicom.read_file(path).pixel_array
#     return img_array[:,:,np.newaxis]
#     # return keras_preprocessing.image.utils.array_to_img(img_array)

data_generator = keras_preprocessing.image.ImageDataGeneratorDicom()
img_from_dir = data_generator.flow_from_directory('/home/francesco/Desktop/Patient02_AIM02/2011-02__Studies', color_mode='grayscale32')

# pydicom.read_file('/home/francesco/Desktop/Patient02_AIM02/2011-02__Studies/Patient02_AIM02_CT_2011-02-06_114007_Tc.multistrato.cranioencefalo_COR.ENC_n91__00000/2.16.840.1.114362.1.11741058.22091046951.529212620.324.153.dcm')

