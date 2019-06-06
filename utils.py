import numpy as np
import scipy.io as scio


def smoothen_curve(curve, smoothing_window_size=800):
    smooth_curve = np.zeros(len(curve))
    for i in range(len(curve)):
        smooth_curve[i] = np.mean(curve[max(0, int(i - smoothing_window_size / 2)):
                                                    min(len(curve), int(i + smoothing_window_size / 2))])
    return smooth_curve


def save_mat_file(path, diction):
    scio.savemat(path, diction)


def read_mat_file(path):
    return scio.loadmat(path)
