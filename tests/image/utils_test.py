import numpy as np
import pytest

from keras_preprocessing.image import utils


def test_validate_filename(tmpdir):
    valid_extensions = ('png', 'jpg')
    filename = tmpdir.ensure('test.png')
    assert utils.validate_filename(str(filename), valid_extensions)

    filename = tmpdir.ensure('test.PnG')
    assert utils.validate_filename(str(filename), valid_extensions)

    filename = tmpdir.ensure('test.some_extension')
    assert not utils.validate_filename(str(filename), valid_extensions)
    assert not utils.validate_filename('some_test_file.png', valid_extensions)

def test_load_img_grayscale32(tmpdir):
    filename_gray16 = str(tmpdir / 'gray16_utils.png')

    original_gray16_array = np.array(500 * np.random.rand(100, 100, 1), dtype=np.int32)
    original_gray16 = utils.array_to_img(original_gray16_array, scale=False)
    assert np.all(original_gray16_array.squeeze() == np.asarray(original_gray16)) 
    original_gray16.save(filename_gray16)
    loaded_img = utils.load_img(filename_gray16, color_mode='grayscale32')
    loaded_img_array = utils.img_to_array(loaded_img)
    
    assert loaded_img_array.shape == original_gray16_array.shape
    assert np.all(loaded_img_array == original_gray16_array)

def test_load_img(tmpdir):
    filename_rgb = str(tmpdir / 'rgb_utils.png')
    filename_rgba = str(tmpdir / 'rgba_utils.png')

    original_rgb_array = np.array(255 * np.random.rand(100, 100, 3),
                                  dtype=np.uint8)
    original_rgb = utils.array_to_img(original_rgb_array, scale=False)
    original_rgb.save(filename_rgb)

    original_rgba_array = np.array(255 * np.random.rand(100, 100, 4),
                                   dtype=np.uint8)
    original_rgba = utils.array_to_img(original_rgba_array, scale=False)
    original_rgba.save(filename_rgba)

    # Test that loaded image is exactly equal to original.

    loaded_im = utils.load_img(filename_rgb)
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == original_rgb_array.shape
    assert np.all(loaded_im_array == original_rgb_array)

    loaded_im = utils.load_img(filename_rgba, color_mode='rgba')
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == original_rgba_array.shape
    assert np.all(loaded_im_array == original_rgba_array)

    loaded_im = utils.load_img(filename_rgb, color_mode='grayscale')
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == (original_rgb_array.shape[0],
                                     original_rgb_array.shape[1], 1)

    # Test that nothing is changed when target size is equal to original.

    loaded_im = utils.load_img(filename_rgb, target_size=(100, 100))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == original_rgb_array.shape
    assert np.all(loaded_im_array == original_rgb_array)

    loaded_im = utils.load_img(filename_rgba, color_mode='rgba',
                               target_size=(100, 100))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == original_rgba_array.shape
    assert np.all(loaded_im_array == original_rgba_array)

    loaded_im = utils.load_img(filename_rgb, color_mode='grayscale',
                               target_size=(100, 100))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == (original_rgba_array.shape[0],
                                     original_rgba_array.shape[1], 1)

    # Test down-sampling with bilinear interpolation.

    loaded_im = utils.load_img(filename_rgb, target_size=(25, 25))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == (25, 25, 3)

    loaded_im = utils.load_img(filename_rgba, color_mode='rgba',
                               target_size=(25, 25))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == (25, 25, 4)

    loaded_im = utils.load_img(filename_rgb, color_mode='grayscale',
                               target_size=(25, 25))
    loaded_im_array = utils.img_to_array(loaded_im)
    assert loaded_im_array.shape == (25, 25, 1)

    # Test down-sampling with nearest neighbor interpolation.

    loaded_im_nearest = utils.load_img(filename_rgb, target_size=(25, 25),
                                       interpolation="nearest")
    loaded_im_array_nearest = utils.img_to_array(loaded_im_nearest)
    assert loaded_im_array_nearest.shape == (25, 25, 3)
    assert np.any(loaded_im_array_nearest != loaded_im_array)

    loaded_im_nearest = utils.load_img(filename_rgba, color_mode='rgba',
                                       target_size=(25, 25),
                                       interpolation="nearest")
    loaded_im_array_nearest = utils.img_to_array(loaded_im_nearest)
    assert loaded_im_array_nearest.shape == (25, 25, 4)
    assert np.any(loaded_im_array_nearest != loaded_im_array)

    # Check that exception is raised if interpolation not supported.

    loaded_im = utils.load_img(filename_rgb, interpolation="unsupported")
    with pytest.raises(ValueError):
        loaded_im = utils.load_img(filename_rgb, target_size=(25, 25),
                                   interpolation="unsupported")


def test_list_pictures(tmpdir):
    filenames = ['test.png', 'test0.jpg', 'test-1.jpeg', '2test.bmp',
                 '2-test.ppm', '3.png', '1.jpeg', 'test.bmp', 'test0.ppm',
                 'test4.tiff', '5-test.tif', 'test.txt', 'foo.csv',
                 'face.gif', 'bar.txt']
    subdirs = ['', 'subdir1', 'subdir2']
    filenames = [tmpdir.ensure(subdir, f) for subdir in subdirs
                 for f in filenames]

    found_images = utils.list_pictures(str(tmpdir))
    assert len(found_images) == 33

    found_images = utils.list_pictures(str(tmpdir), ext='png')
    assert len(found_images) == 6


def test_array_to_img_and_img_to_array():
    height, width = 10, 8

    # Test the data format
    # Test RGB 3D
    x = np.random.random((3, height, width))
    img = utils.array_to_img(x, data_format='channels_first')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_first')
    assert x.shape == (3, height, width)

    # Test RGBA 3D
    x = np.random.random((4, height, width))
    img = utils.array_to_img(x, data_format='channels_first')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_first')
    assert x.shape == (4, height, width)

    # Test 2D
    x = np.random.random((1, height, width))
    img = utils.array_to_img(x, data_format='channels_first')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_first')
    assert x.shape == (1, height, width)

    # Test tf data format
    # Test RGB 3D
    x = np.random.random((height, width, 3))
    img = utils.array_to_img(x, data_format='channels_last')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_last')
    assert x.shape == (height, width, 3)

    # Test RGBA 3D
    x = np.random.random((height, width, 4))
    img = utils.array_to_img(x, data_format='channels_last')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_last')
    assert x.shape == (height, width, 4)

    # Test 2D
    x = np.random.random((height, width, 1))
    img = utils.array_to_img(x, data_format='channels_last')
    assert img.size == (width, height)

    x = utils.img_to_array(img, data_format='channels_last')
    assert x.shape == (height, width, 1)

    # Test invalid use case
    with pytest.raises(ValueError):
        x = np.random.random((height, width))  # not 3D
        img = utils.array_to_img(x, data_format='channels_first')

    with pytest.raises(ValueError):
        x = np.random.random((height, width, 3))
        # unknown data_format
        img = utils.array_to_img(x, data_format='channels')

    with pytest.raises(ValueError):
        # neither RGB, RGBA, or gray-scale
        x = np.random.random((height, width, 5))
        img = utils.array_to_img(x, data_format='channels_last')

    with pytest.raises(ValueError):
        x = np.random.random((height, width, 3))
        # unknown data_format
        img = utils.img_to_array(x, data_format='channels')

    with pytest.raises(ValueError):
        # neither RGB, RGBA, or gray-scale
        x = np.random.random((height, width, 5, 3))
        img = utils.img_to_array(x, data_format='channels_last')


if __name__ == '__main__':
    pytest.main([__file__])
