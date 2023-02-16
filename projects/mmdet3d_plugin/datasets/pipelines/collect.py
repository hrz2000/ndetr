from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import PIPELINES
import mmcv
import numpy as np
from projects.configs.detr3d.new.cc import on_cc, enable_mc

if on_cc and enable_mc:
    from petrel_client.client import Client
    conf_path = '~/petreloss.conf'
    #conf_path = '~/.s3cfg'
    client = Client(conf_path)
    print("create client")

def load_img(img_url):
    img_bytes = client.get(img_url)
    img = mmcv.imfrombytes(img_bytes)
    return img

class LoadImageFromCeph:
    """Load an image from file.
    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='petrel')``.
    """

    def __init__(self, file_client_args=dict(backend='petrel')):
        self.file_client_args = file_client_args.copy()

    def __call__(self, img_path):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        self.file_client = mmcv.FileClient(**self.file_client_args)

        img_bytes = self.file_client.get(img_path)
        img = mmcv.imfrombytes(img_bytes)

        return img

@PIPELINES.register_module()
class Collect3Dx(object):
    def __init__(
        self,
        keys):
        self.keys = keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in results:
            if key not in self.keys:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        # self.load_img = LoadImageFromCeph()

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        if on_cc == False or enable_mc == False:
            img = np.stack([mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        else:
            img = np.stack([load_img(name) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str