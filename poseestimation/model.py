import cv2
import numpy as np
import urllib.request
import shutil
from keras.models import load_model
from os.path import join, isfile
from scipy.ndimage.filters import gaussian_filter
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
import math

# -----------------------------------
# neural network part
# -----------------------------------

def relu(x):
    return Activation('relu')(x)


def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def vgg_block(x):
    # Block 1
    x = conv(x, 64, 3, "conv1_1")
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1")
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2")
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1")
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2")
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3")
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4")
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1")
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2")
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM")
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM")
    x = relu(x)

    return x


def stage1_block(x, num_p, branch):
    # Block 1
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)

    return x


def stageT_block(x, num_p, stage, branch):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))

    return x

# -----------------------------------
# neural network part
# -----------------------------------


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

        modelf = join(model_dir, 'model_weights.h5')
        if not isfile(modelf):
            url = 'http://188.138.127.15:81/models/model_weights.h5'
            with urllib.request.urlopen(url) as res, open(modelf, 'wb') as f:
                shutil.copyfileobj(res, f)

        input_shape = (None, None, 3)

        img_input = Input(shape=input_shape)

        stages = 6
        np_branch1 = 38
        np_branch2 = 19

        img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

        # VGG
        stage0_out = vgg_block(img_normalized)

        # stage 1
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
        model.load_weights(modelf)
        self.model = model

        # modelf = join(model_dir, 'poseestimation.h5')
        # if not isfile(modelf):
        #     url = 'http://188.138.127.15:81/models/poseestimation.h5'
        #     with urllib.request.urlopen(url) as res, open(modelf, 'wb') as f:
        #         shutil.copyfileobj(res, f)
        # self.model = load_model(modelf)

    def predict(self, X):
        """
        end-to-end prediction
        :param X:
        :return:
        """
        if len(X.shape) == 3:  # single image: make it big
            X = np.expand_dims(X, 0)
        n, h, w, _ = X.shape
        thre1 = self.peak_threshold

        # get the heatmaps and pafs for all scales for each
        # entry in X
        heatmaps, pafs = self.predict_pafs_and_heatmaps(X)

        POSITIONS = []

        for cur_frame in range(n):
            all_peaks = []
            peak_counter = 0
            for part in range(19-1):
                hm = heatmaps[cur_frame][:,:,part]
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

            # -----------------------
            mid_num = 10
            paf_avg = np.squeeze(pafs[cur_frame])
            thre2 = 0.05
            special_k = []
            connection_all = []

            limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                       [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                       [1, 16], [16, 18], [3, 17], [6, 18]]

            mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                      [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                      [55, 56], [37, 38], [45, 46]]

            for k in range(len(mapIdx)):
                score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
                candA = all_peaks[limbSeq[k][0] - 1]
                candB = all_peaks[limbSeq[k][1] - 1]
                #indexA, indexB = limbSeq[k]
                nA = len(candA)
                nB = len(candB)
                if nA != 0 and nB != 0:
                    connection_candidate = []
                    for i in range(nA):
                        for j in range(nB):
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                            if norm == 0:
                                continue
                            vec = np.divide(vec, norm)

                            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                              for I in range(len(startend))])
                            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                              for I in range(len(startend))])

                            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * h / norm - 1, 0)
                            criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                            criterion2 = score_with_dist_prior > 0

                            if criterion1 and criterion2:
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                    connection = np.zeros((0, 5))
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        if (i not in connection[:, 3] and j not in connection[:, 4]):
                            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # last number in each row is the total parts number of that person
            # the second last number in each row is the score of the overall configuration
            subset = -1 * np.ones((0, 20))
            candidate = np.array([item for sublist in all_peaks for item in sublist])

            for k in range(len(mapIdx)):
                if k not in special_k:
                    partAs = connection_all[k][:, 0]
                    partBs = connection_all[k][:, 1]
                    indexA, indexB = np.array(limbSeq[k]) - 1

                    for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):  # 1:size(subset,1):
                            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                                subset_idx[found] = j
                                found += 1

                        if found == 1:
                            j = subset_idx[0]
                            if (subset[j][indexB] != partBs[i]):
                                subset[j][indexB] = partBs[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        elif found == 2:  # if found 2 and disjoint, merge them
                            j1, j2 = subset_idx
                            print("found = 2")
                            membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                            if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)
                            else:  # as like found == 1
                                subset[j1][indexB] = partBs[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < 17:
                            row = -1 * np.ones(20)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            row[-1] = 2
                            row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                            subset = np.vstack([subset, row])

            # delete some rows of subset which has few parts occur
            deleteIdx = []
            for i in range(len(subset)):
                if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                    deleteIdx.append(i)
            subset = np.delete(subset, deleteIdx, axis=0)

            # POSITIONS -> (n, #persons, 17)
            nbr_persons = len(subset)
            cur_positions = []
            for _ in range(nbr_persons):
                cur_positions.append([(-1, -1)] * 18)

            for j in range(17):
                for k in range(nbr_persons):
                    index = subset[k][np.array(limbSeq[j]) - 1]
                    if -1 in index:
                        continue

                    # joint position:
                    a, b = np.array(limbSeq[j]) - 1
                    _Y = candidate[index.astype(int), 0]
                    _X = candidate[index.astype(int), 1]
                    cur_positions[k][a] = (_X[0], _Y[0])
                    cur_positions[k][b] = (_X[1], _Y[1])

            POSITIONS.append(np.array(cur_positions))

        # ------------------------

        return POSITIONS

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
        Heatmaps = np.zeros((n, h, w, 19))
        Pafs = np.zeros((n, h, w, 38))

        for j, scale in enumerate(multiplier):
            w_res = int(w * scale); h_res = int(h * scale)
            pad_down = 0 if (h_res % stride == 0) else stride - (h_res % stride)
            pad_right = 0 if (w_res % stride == 0) else stride - (w_res % stride)
            X_resized = np.zeros((n, h_res+pad_down, w_res+pad_right, 3), 'uint8')
            for i in range(n):
                I_res = cv2.resize(X[i], (w_res, h_res), interpolation=cv2.INTER_CUBIC)
                I_res, pad = padRightDownCorner(I_res, stride, padValue)
                X_resized[i] = I_res

            pafs, heatmaps = self.model.predict(X_resized)

            for i in range(n):
                hm = cv2.resize(heatmaps[i], (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                hm = cv2.resize(hm[0:h_res,0:w_res], (w,h), interpolation=cv2.INTER_CUBIC)
                Heatmaps[i] = Heatmaps[i] + hm / nbr_scales

                paf = cv2.resize(pafs[i], (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                paf = cv2.resize(paf[0:h_res,0:w_res], (w, h), interpolation=cv2.INTER_CUBIC)
                Pafs[i] = Pafs[i] + paf / nbr_scales


        return Heatmaps, Pafs
