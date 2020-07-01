import sys
from pathlib import Path
import subprocess as sp
from math import copysign
import json
from datetime import datetime, timedelta
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import vtf_config as c

def create_path(filepath, outputpath):
    filename = Path(filepath).stem
    tmp = Path(outputpath).joinpath(filename)
    tmp.mkdir(parents = True, exist_ok = True)
    files = list(tmp.glob('*.jpg'))
    if len(files):
        last_keyframe = int(files[-1].stem[-3:]) + 1
        
    else:
        last_keyframe = 1
        
    return filename, str(tmp), last_keyframe


def concat_path(path1, path2):
    return str(Path(path1).joinpath(path2))


def time_difference(start, end):
    ts = start.split(':')
    time1 = datetime(100, 1, 1, int(ts[0]), int(ts[1]), int(ts[2]))
    ts = end.split(':')
    time2 = datetime(100, 1, 1, int(ts[0]), int(ts[1]), int(ts[2]))
    time_dif = time2 - time1
    if time_dif.seconds <= 0:
        sys.exit('End time must be higher than start time')
    return time_dif.seconds


def create_mask():
    # create a mask for goodpointstotrack
    yi = int(c.out_H * c.overlap)
    xi = int(c.out_W * c.overlap)
    mask = np.zeros((c.out_H, c.out_W), dtype = np.uint8)
    mask[yi:-yi,xi:-xi] = 255
    return mask


def get_info(file):
    path = Path(file)
    if not path.exists():
        sys.exit('Video file not found')

    ffprobe_cmd = ['ffprobe', '-v', 'error',
                   '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,r_frame_rate:stream_tags=timecode',
                   '-of', 'json',
                   '-i', file]

    out_raw = sp.check_output(ffprobe_cmd)
    out_dict = json.loads(out_raw)

    # define variables
    frame_W = out_dict['streams'][0]['width']
    frame_H = out_dict['streams'][0]['height']

    fps_s = out_dict['streams'][0]['r_frame_rate'].split('/')
    fps = float(fps_s[0]) / float(fps_s[1])

    if out_dict['streams'][0]['tags'] == {}:
        time_real_start = None
    else:
        ts = out_dict['streams'][0]['tags']['timecode'].split(':')
        time_real_start =  datetime(100, 1, 1, int(ts[0]), int(ts[1]), int(ts[2]))
    
    return frame_H, frame_W, fps, time_real_start


# open file with ffmpeg and create a pipe
class Capture:
    """
    A custom class to read every frame of a a file using ffmpeg
    similar to cv2.VideoCapture(), but this class supports video filters and
    a easy and accurate way to seek a video to a specific time
    """
    def __init__(self, file, time_start, time_dif, start_real):
        self.frame_H, self.frame_W, self.fps, start_real_file = get_info(file)
        self.npixels = self.frame_H * self.frame_W * 3
        self.t_add = 1 / self.fps

        ts = time_start.split(':')
        self.time_start_dt = timedelta(hours = int(ts[0]), minutes = int(ts[1]), seconds = int(ts[2]))

        # get real time, based on the time start of the video
        if start_real is None:
            if start_real_file is None:
                sys.exit('Could not get timestamp from video file')
            else:
                self.start_real = start_real_file + self.time_start_dt + timedelta(seconds = c.time_offset)
        else:
            ts = start_real.split(':')
            self.start_real = datetime(100, 1, 1, int(ts[0]), int(ts[1]), int(ts[2]))

        # open movie file
        ffmpeg_cmd = ['ffmpeg', '-v', 'quiet',
                      '-ss', time_start,
                      '-t', str(time_dif),
                      '-i', file,
                      '-vf', 'pp=ci|a,hqdn3d',  # deinterlace and denoise video
                      '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
                      '-vcodec', 'rawvideo',
                      '-an', '-sn',           	# disable audio and sub-title processing
                      '-f', 'image2pipe', '-']    
        self.pipe = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, bufsize = 10**7)
        self.raw = self.pipe.stdout.read(self.npixels)
        self.counter = -1

    # grab next frame
    def retrieve(self):
        image =  np.frombuffer(self.raw, dtype='uint8')
        image = image.reshape((self.frame_H, self.frame_W, 3))
        self.raw = self.pipe.stdout.read(self.npixels)
        self.counter += 1
        return image[:,:,:]

    # check if ffmpeg is still runing
    def ret(self):
        if self.raw == b'':
            return False
        else:
            return True
    
    # skip n frames 
    def skip(self):
        self.counter += c.blur_skip
        for _ in range(c.blur_skip):
            self.raw = self.pipe.stdout.read(self.npixels)

    # get real time
    def real_time(self):
        times = self.counter * self.t_add
        time = self.start_real + timedelta(seconds = times)
        return time.strftime("%H:%M:%S")
    
    # get elapsed time in seconds
    def time(self):
        times = self.counter * self.t_add
        time = self.time_start_dt + timedelta(seconds = times)
        hours, rem = divmod(time.seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'


class CenterImg:
    def __init__(self, cap):
        frame_H_new = cap.frame_H - c.overlay_top
        self.x_down = cap.frame_W // c.downsize_interpolate
        self.y_down = frame_H_new // c.downsize_interpolate
        self.x = np.arange(self.x_down)
        self.y = np.arange(self.y_down)
        x_cut = (cap.frame_W - c.out_W) // (2*c.downsize_interpolate)
        y_cut = (frame_H_new - c.out_H) // (2*c.downsize_interpolate)
        x2_down = self.x_down // 2
        y2_down = self.y_down // 2
        self.x_predict = self.x[(x2_down - x_cut):(x2_down + x_cut + 1)]
        self.y_predict = self.y[(y2_down - y_cut):(y2_down + y_cut + 1)]
        self.x_crop = c.out_W // 2
        self.y_crop = c.out_H // 2

    def process(self, img, old_ed):
        # get the center of the image with more brightness, constrained by out_W and out_H 
        resized = cv2.resize(img, dsize=(self.x_down, self.y_down), interpolation = cv2.INTER_AREA)
        
        # interpolate intensity on the downsized image
        ss = self.x_down*self.y_down*resized.var()
        fit = RectBivariateSpline(self.y, self.x, resized, kx = 2, ky = 2, s = ss)
        predict = fit(self.y_predict, self.x_predict)
        
        # get coordinates with highest brightness, on the original image
        maxLoc = np.unravel_index(np.argmax(predict), predict.shape)
        x_final = (maxLoc[1] + self.x_predict[0])*c.downsize_interpolate - self.x_crop
        y_final = (maxLoc[0] + self.y_predict[0])*c.downsize_interpolate - self.y_crop

        if old_ed is not None:
            x_diff1 = x_final - old_ed[2]
            x_diff2 = min(c.key_move_thr, abs(x_diff1))
            x_final = old_ed[2] + int(copysign(x_diff2, x_diff1))

            y_diff1 = y_final - old_ed[0]
            y_diff2 = min(c.key_move_thr, abs(y_diff1))
            y_final = old_ed[0] + int(copysign(y_diff2, y_diff1))
        
        # return edges of the original image
        return (y_final, y_final + c.out_H, x_final, x_final + c.out_W)


# get the mean, excluding % lowest and highest values
def mean_interval(img, n_percent):
    resized_max = np.partition(img.ravel(),  -n_percent)[:-n_percent]
    resized_min = np.partition(resized_max,  n_percent)[n_percent:]
    return resized_min.mean()


a_sobel_cor = -1 / 150
b_sobel_cor = c.rectify_flow_max - (a_sobel_cor*50)
def sobel_cor(lap_mean):
    y = lap_mean*a_sobel_cor + b_sobel_cor
    return min(max(1.0, y), c.rectify_flow_max)


# normalize the image based on mean and variance
def normalize(img):
    a = c.norm_var / img.var()
    a = max(min(a, c.max_contrast), c.min_contrast)
    b = 127 - a * img.mean()
    normalized = cv2.convertScaleAbs(img, alpha = a, beta = b)
    return normalized


# skip and display N frames descarted by blurness
def skip_n_display(plot, cap, img, lap_max):
    # process and display next n consecutive frames
    plot.update(lap_max)
    cv2.putText(img, 'Frame descaterted by blurriness', **c.text_params)
    cv2.imshow('original frame', cv2.resize(img, **c.resize_params))
    cv2.waitKey(1)
    for _ in range(c.blur_skip):
        img = cap.retrieve()
        gray = img[c.overlay_top:,:,1]
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=5))
        plot.update(lap.max())
        cv2.putText(img,'Frame descaterted by blurriness', **c.text_params)
        cv2.imshow('original frame', cv2.resize(img, **c.resize_params))
        cv2.waitKey(1)


class Plot:
    def __init__(self, fps):
        plot_n = c.window_size*fps
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget(size = (600,250), title='Focus Detector')
        self.p1_data = np.full(plot_n, c.blur_thr, float)
        self.ptr1 = 0

        self.p1 = self.win.addPlot()
        self.curve1 = self.p1.plot(pen='r')
        self.p1.hideButtons()
        self.inf1 = pg.InfiniteLine(pos=c.blur_thr, angle=0, pen='y')
        self.p1.addItem(self.inf1)
        self.win.show()
        
    def update(self, y1_data):
        self.p1_data[:-1] = self.p1_data[1:]
        self.p1_data[-1] = y1_data
        
        self.curve1.setData(y=self.p1_data)

        self.ptr1 += 1
        self.curve1.setPos(self.ptr1, 0)
        QtGui.QApplication.processEvents()

    def close(self):
        self.win.close()
        self.app.exit()
