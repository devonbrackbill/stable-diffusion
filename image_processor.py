import cv2
import os
import potrace
from matplotlib.pyplot import imshow, show
import numpy as np
from typing import Callable, List


class ImageProcessor:
    '''
    Usage:
    from image_processor import ImageProcessor
    ip = ImageProcessor('image.png', 'outputdir')
    # define arbitrary threshold function
    def quantile75(x):
        return np.quantile(x, 0.75)
    i.process_image(threshold_funcs=[np.mean, np.median, quantile75])
    '''
    def __init__(self, imagepath: str, outdir: str = ''):
        self.img_raw = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
        self.img_thresholded = None
        self.img_potrace_trace = None
        self.imagepath = imagepath
        # get the filename from the imagepath
        self.filename = os.path.basename(imagepath).split('.')[0]
        self.outdir = outdir
        self.filename_postfix = None
        self.outputfile = self.refresh_outputfile()
        self._check_image_size()
        self._setup_directory()
        self._print_status()

    def refresh_outputfile(self):
        '''refresh outputfile'''
        self.outputfile = os.path.join(
            self.outdir, self.filename +
            ('-' + self.filename_postfix if self.filename_postfix else '') +
            '.svg')

    def _check_image_size(self):
        '''check image size'''
        m = 'Image is size {}, but should be (512, 512, 3)'.\
            format(self.img_raw.shape)
        assert self.img_raw.shape == (512, 512, 3), m

    def _print_status(self):
        '''print status'''
        print('ImageProcessor initialized with image {}'.
              format(self.imagepath))
        print('Filename is {}'.format(self.filename))
        print('Output directory is {}'.format(self.outdir))
        print('Output file is {}'.format(self.outputfile))
        print('Image size is {}'.format(self.img_raw.shape))

    def _setup_directory(self):
        '''setup directory'''
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        print('Created directory {}'.format(self.outdir))

    def show_image(self, version: str = 'raw'):
        '''show image
        Args:
            version (str): 'raw', 'thresholded', or 'final'
        '''
        if version == 'raw':
            imshow(self.img_raw)
        elif version == 'thresholded':
            imshow(self.img_thresholded)
        elif version == 'final':
            imshow(self.img)
        else:
            raise NotImplementedError('Unknown version {}'.format(version))
        show()

    def color_to_bw(self, img: np.array, threshold_func: Callable,
                    greaterthan_to_black: bool = True, *args, **kwargs):
        '''convert image to black and white
        Args:
            threshold_func (function): threshold function
            greaterthan_to_black (bool): if True, pixels greater than threshold
                are set to black, otherwise to white
        '''
        threshold_value = threshold_func(img, *args, **kwargs)
        if greaterthan_to_black:
            img_thresholded = np.where(img <= threshold_value, 255, 0)
        else:
            img_thresholded = np.where(img <= threshold_value, 0, 255)
        max_pixelwise = \
            np.maximum(
                np.maximum(img_thresholded[:, :, 0], img_thresholded[:, :, 1]),
                img_thresholded[:, :, 0], img_thresholded[:, :, 2])
        # set pixels in each channel/dimension to the max_pixelwise value
        img_thresholded[:, :, 0] = max_pixelwise
        img_thresholded[:, :, 1] = max_pixelwise
        img_thresholded[:, :, 2] = max_pixelwise
        return img_thresholded
    
    def resize_image(self, img: np.array, size: int = None):
        '''resize image; only allows square images
        Args:
            img (np.array): image
            size (tuple): size of the image
        '''
        if size is None:
            return img
        else:
            return cv2.resize(img, (size, size))

    def png_to_svg(self):
        '''convert png to svg'''
        # potrace.Bitmap takes 1 channel images, and all channels are the same
        bitmap = potrace.Bitmap(self.img_thresholded[:, :, 0])
        self.img_potrace_trace = bitmap.trace()

    def save_svg(self):
        '''save svg to outputfile'''
        with open(self.outputfile, "w") as f:
            f.write(
                '<svg version="1.1"' +
                ' xmlns="http://www.w3.org/2000/svg"' +
                ' xmlns:xlink="http://www.w3.org/1999/xlink"' +
                ' width="%d" height="%d"' % (512, 512) +
                ' viewBox="0 0 %d %d">' % (512, 512) +
                ' <!--! IconAI by @iconai https://iconai.com Copyright' +
                ' IconAI LLC-->'
            )
            parts = []  # parts of the svg file
            # add the path to the parts
            for curve in self.img_potrace_trace:
                fs = curve.start_point
                parts.append("M%f,%f" % (fs.x, fs.y))
                for segment in curve.segments:
                    if segment.is_corner:
                        a = segment.c
                        parts.append("L%f,%f" % (a.x, a.y))
                        b = segment.end_point
                        parts.append("L%f,%f" % (b.x, b.y))
                    else:
                        a = segment.c1
                        b = segment.c2
                        c = segment.end_point
                        parts.append("C%f,%f %f,%f %f,%f" % (a.x, a.y, b.x,
                                                             b.y, c.x, c.y))
                parts.append("z")
            f.write(
                '<path stroke="none" fill="%s" fill-rule="evenodd" d="%s"/>'
                % ("black", "".join(parts))
            )
            f.write("</svg>")
        # get file size in bytes
        filesize = os.path.getsize(self.outputfile)
        print("Saved svg to {} ({} bytes)".format(self.outputfile, filesize))

    def process_image(self, threshold_funcs: List[Callable],
                      greaterthan_to_blacks: List[bool] = None,
                      *args, **kwargs):
        '''process image
        Args:
            threshold_func (function): threshold function
        '''
        if greaterthan_to_blacks is None:
            greaterthan_to_blacks = [True] * len(threshold_funcs)
        for threshold_func, greaterthan_to_black in zip(threshold_funcs,
                                                        greaterthan_to_blacks):
            self.img_thresholded = \
                self.color_to_bw(self.img_raw, threshold_func, *args, **kwargs)
            self.img_resized = \
                self.resize_image(self.img_thresholded, size=None)

            self.png_to_svg()
            self.filename_postfix = threshold_func.__name__ + '-' + \
                str(greaterthan_to_black)
            self.refresh_outputfile()
            self.save_svg()
        print('Processed image {}'.format(self.imagepath))


def quantile25(x):
    return np.quantile(x, 0.25)


def parallel_helper(args):
    '''helper function for parallel processing'''
    path_to_file, outputdir = args
    ip = ImageProcessor(path_to_file, outputdir)
    ip.process_image(threshold_funcs=[np.mean, np.mean],
                     greaterthan_to_blacks=[True, False])
    return ip.outputfile
