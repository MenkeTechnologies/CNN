import numpy as np

from cnn_comparison.util.utils import *


class ImageFileToNumPyAry:
    def __init__(self, filename, array224, array299, array331):
        self.filename = filename
        self.__dim_map = {
            D_224: array224,
            D_229: array299,
            D_331: array331
        }

    def img_for_dim(self, dim):
        return self.__dim_map[dim]


class TimeResults:
    def __init__(self):
        self.__times = []

    def add_time(self, time):
        self.__times.append(time)

    def get_time_ms_avg(self):
        tot = 0
        for t in self.__times:
            tot += t

        return to_ms(tot / len(self.__times))

    def get_time_ns_avg(self):
        tot = 0
        for t in self.__times:
            tot += t

        return tot / len(self.__times)


class PredictionResults:
    def __init__(self):
        self.__map_sum = {}
        self.__map_cnt = {}

    def add_prediction(self, label, prob):
        tot_prob = self.__map_sum.get(label, 0)
        cnt = self.__map_cnt.get(label, 0)
        self.__map_sum[label] = tot_prob + prob
        self.__map_cnt[label] = cnt + 1

    def get_label_avg(self):
        map_avg = {}
        for label, sum in self.__map_sum.items():
            map_avg[label] = sum / self.__map_cnt[label]

        return map_avg

    def get_label_highest(self):
        max_label = ""
        max_prob_sum = 0
        for label, sum in self.__map_sum.items():
            if sum > max_prob_sum:
                max_label = label
                max_prob_sum = sum

        label_ = max_prob_sum / self.__map_cnt[max_label]
        return max_label, f"{label_:.2f}"

    def get_iter(self, label):
        return self.__map_cnt[label]


class ResultsWrapper:
    def __init__(self):
        self.__results_time = TimeResults()

        self.__results = {}
        for img in TEST_IMG_FILENAMES:
            self.__results[img] = PredictionResults()

    def for_img(self, image):
        return self.__results[image]

    def time_results(self):
        return self.__results_time


class ModelWrapper:
    def __init__(self, model, processor, decode, dim):
        self.model = model
        self.processor = processor
        self.decode = decode
        self.processed = None
        self.to_process = None
        self.dim = dim
        self.results = ResultsWrapper()

    def process_single(self, images=None):
        if self.to_process is None:
            self.processed = self.processor(np.expand_dims(images, axis=0))
        else:
            self.processed = self.processor(np.expand_dims(self.to_process, axis=0))

    def process_mult(self, images=None):
        if self.to_process is None:
            self.processed = self.processor(np.array(images))
        else:
            self.processed = self.processor(np.array(self.to_process))
