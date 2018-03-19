import cv2
import numpy as np
import urllib.request
import shutil
from keras.models import load_model
from os.path import join, isfile
from scipy.ndimage.filters import gaussian_filter


def padRightDownCorner(img, stride, padValue):
    """
    Taken from:
    https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/util.py
    :param img:
    :param stride:
    :param padValue:
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


class PoseEstimator:

    def __init__(self,
                 model_dir='/tmp',
                 boxsize=368,
                 padValue=128,
                 stride=8,
                 scale_search=[0.5, 1, 1.5, 2],
                 peak_threshold=0.1):
        self.boxsize = boxsize
        self.padValue = padValue
        self.stride = stride
        self.scale_search = scale_search
        self.peak_threshold = peak_threshold

        modelf = join(model_dir, 'poseestimation.h5')
        if not isfile(modelf):
            url = 'http://188.138.127.15:81/models/poseestimation.h5'
            with urllib.request.urlopen(url) as res, open(modelf, 'wb') as f:
                shutil.copyfileobj(res, f)

        self.model = load_model(modelf)

    def predict(self, X):
        """
        end-to-end prediction
        :param X:
        :return:
        """
        if len(X.shape) == 3:  # single image: make it big
            X = np.expand_dims(X, 0)
        n = X.shape[0]
        thre1 = self.peak_threshold

        # get the heatmaps and pafs for all scales for each
        # entry in X
        heatmaps, pafs = self.predict_pafs_and_heatmaps(X)

        all_peaks = []
        peak_counter = 0

        for i in range(n):
            for part in range(19-1):
                hm = heatmaps[i][:,:,part]
                blur = gaussian_filter(hm, sigma=3)  #TODO use cv2 here..

                map_left = np.zeros(blur.shape)
                map_left[1:, :] = blur[:-1, :]
                map_right = np.zeros(blur.shape)
                map_right[:-1, :] = blur[1:, :]
                map_up = np.zeros(blur.shape)
                map_up[:, 1:] = blur[:, :-1]
                map_down = np.zeros(blur.shape)
                map_down[:, :-1] = blur[:, 1:]

                peaks_binary = np.logical_and.reduce(
                    (blur >= map_left, blur >= map_right,
                     blur >= map_up, blur >= map_down, blur > thre1))

                peaks = list(
                    zip(np.nonzero(peaks_binary)[1],
                        np.nonzero(peaks_binary)[0]))

                peaks_with_score = [x + (hm[x[1], x[0]],) for x in peaks]
                id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)

        return all_peaks


    def predict_pafs_and_heatmaps(self, X):
        """

        :param X:
        :return:
        """
        if len(X.shape) == 3:  # single image: make it big
            X = np.expand_dims(X, 0)

        boxsize = self.boxsize
        scale_search = self.scale_search
        stride = self.stride
        padValue = self.padValue
        n, h, w, _ = X.shape

        multiplier = [x * boxsize / h for x in scale_search]

        nbr_scales = len(scale_search)
        heatmaps_over_scales = np.zeros((n, nbr_scales, h, w, 19))
        pafs_over_scales = np.zeros((n, nbr_scales, h, w, 38))

        for j, scale in enumerate(multiplier):
            w_res = int(w * scale); h_res = int(h * scale)
            pad_down = 0 if (h_res % stride == 0) else stride - (h_res % stride)
            pad_right = 0 if (w_res % stride == 0) else stride - (w_res % stride)
            X_resized = np.zeros((n, h_res+pad_down, w_res+pad_right, 3), 'uint8')
            for i in range(n):
                I_res = cv2.resize(X[i], (w_res, h_res), interpolation=cv2.INTER_CUBIC)
                I_res, pad = padRightDownCorner(I_res, stride, padValue)
                X_resized[i] = I_res

            # the required shape is (n,w,h,3) but atm it is (n,h,w,3)
            X_resized = np.swapaxes(X_resized, 1, 2)

            pafs, heatmaps = self.model.predict(X_resized)

            # 'restore' the old order -> h,w
            heatmaps = np.swapaxes(heatmaps, 1, 2)
            pafs = np.swapaxes(pafs, 1, 2)

            for i in range(n):
                hm = cv2.resize(heatmaps[i], (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                hm = cv2.resize(hm[0:h,0:w], (w,h), interpolation=cv2.INTER_CUBIC)
                heatmaps_over_scales[i, j] = hm

                paf = cv2.resize(pafs[i], (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                paf = cv2.resize(paf[0:h,0:w], (w, h), interpolation=cv2.INTER_CUBIC)
                pafs_over_scales[i,j] = paf


        return np.mean(heatmaps_over_scales, axis=1), \
               np.mean(pafs_over_scales, axis=1)
