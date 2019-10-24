import os
import warnings
import multiprocessing

import numpy as np
import pydicom

from .iterator import Iterator
from .directory_iterator import DirectoryIterator
from .utils import _PIL_INTERPOLATION_METHODS, array_to_img, img_to_array, _list_valid_filenames_in_directory
from .iterator import BatchFromFilesMixin
from .image_data_generator import ImageDataGenerator


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    x = pydicom.read_file(path)
    img = array_to_img(x.pixel_array)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'grayscale32':
        if img.mode != 'I':
            img = img.convert('I')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", "grayscale32" or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


class DicomIterator(Iterator):
    white_list_formats = ('dcm')
    def __init__(self, n, batch_size, shuffle, seed):
        Iterator.__init__(self, n, batch_size, shuffle, seed)
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        return batch_x, batch_y, self.sample_weight[index_array]

class DirectoryIteratorDicom_improved(DicomIterator, DirectoryIterator):
    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        DirectoryIterator.__init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32')

class DirectoryIteratorDicom(BatchFromFilesMixin, DicomIterator):
    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(DirectoryIteratorDicom, self).set_processing_attrs(image_data_generator,
                                                            target_size,
                                                            color_mode,
                                                            data_format,
                                                            save_to_dir,
                                                            save_prefix,
                                                            save_format,
                                                            subset,
                                                            interpolation)
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super(DirectoryIteratorDicom, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

class ImageDataGeneratorDicom(ImageDataGenerator):
    def __init__(self):
            ImageDataGenerator.__init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 dtype='float32'):


    def flow_from_directory(self,
                        directory,
                        target_size=(256, 256),
                        color_mode='rgb',
                        classes=None,
                        class_mode='categorical',
                        batch_size=32,
                        shuffle=True,
                        seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='png',
                        follow_links=False,
                        subset=None,
                        interpolation='nearest'):
    """Takes the path to a directory & generates batches of augmented data.

    # Arguments
        directory: string, path to the target directory.
            It should contain one subdirectory per class.
            Any PNG, JPG, BMP, PPM or TIF images
            inside each of the subdirectories directory tree
            will be included in the generator.
            See [this script](
            https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
            for more details.
        target_size: Tuple of integers `(height, width)`,
            default: `(256, 256)`.
            The dimensions to which all images found will be resized.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            Whether the images will be converted to
            have 1, 3, or 4 channels.
        classes: Optional list of class subdirectories
            (e.g. `['dogs', 'cats']`). Default: None.
            If not provided, the list of classes will be automatically
            inferred from the subdirectory names/structure
            under `directory`, where each subdirectory will
            be treated as a different class
            (and the order of the classes, which will map to the label
            indices, will be alphanumeric).
            The dictionary containing the mapping from class names to class
            indices can be obtained via the attribute `class_indices`.
        class_mode: One of "categorical", "binary", "sparse",
            "input", or None. Default: "categorical".
            Determines the type of label arrays that are returned:
            - "categorical" will be 2D one-hot encoded labels,
            - "binary" will be 1D binary labels,
                "sparse" will be 1D integer labels,
            - "input" will be images identical
                to input images (mainly used to work with autoencoders).
            - If None, no labels are returned
                (the generator will only yield batches of image data,
                which is useful to use with `model.predict_generator()`).
                Please note that in case of class_mode None,
                the data still needs to reside in a subdirectory
                of `directory` for it to work correctly.
        batch_size: Size of the batches of data (default: 32).
        shuffle: Whether to shuffle the data (default: True)
            If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        save_to_dir: None or str (default: None).
            This allows you to optionally specify
            a directory to which to save
            the augmented pictures being generated
            (useful for visualizing what you are doing).
        save_prefix: Str. Prefix to use for filenames of saved pictures
            (only relevant if `save_to_dir` is set).
        save_format: One of "png", "jpeg"
            (only relevant if `save_to_dir` is set). Default: "png".
        follow_links: Whether to follow symlinks inside
            class subdirectories (default: False).
        subset: Subset of data (`"training"` or `"validation"`) if
            `validation_split` is set in `ImageDataGenerator`.
        interpolation: Interpolation method used to
            resample the image if the
            target size is different from that of the loaded image.
            Supported methods are `"nearest"`, `"bilinear"`,
            and `"bicubic"`.
            If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
            supported. If PIL version 3.4.0 or newer is installed,
            `"box"` and `"hamming"` are also supported.
            By default, `"nearest"` is used.

    # Returns
        A `DirectoryIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
    """
    return DirectoryIteratorDicom_improved(
        directory,
        self,
        target_size=target_size,
        color_mode=color_mode,
        classes=classes,
        class_mode=class_mode,
        data_format=self.data_format,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        save_format=save_format,
        follow_links=follow_links,
        subset=subset,
        interpolation=interpolation
    )