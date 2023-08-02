import json
import logging

import numpy as np
import os

from . import teaser
from . import correspondence_helpers
from . import ransac


def register(source_npz: str, target_npz: str, algorithm: str = 'TEASER',
             config_path: str = '', n_keypoints: int = 5000):

    if config_path == '':
        config_path = os.path.dirname(os.path.abspath(__file__)) + '/configs/' + algorithm + '.json'
    config_file = open(config_path)
    config = json.load(config_file)

    # create logger
    # os.makedirs('logs', exist_ok=True)
    # logname = 'logs/' + 'test' + '.log'
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    # logging.basicConfig(filename=logname, level=logging.DEBUG, filemode='w')

    target_npz = np.load(target_npz)
    source_npz = np.load(source_npz)
    target_features = target_npz['features']
    source_features = source_npz['features']
    target_xyz = target_npz['xyz_down'].T
    source_xyz = source_npz['xyz_down'].T

    print("Source features: {}".format(source_features.shape))
    print("Target features: {}".format(target_features.shape))

    # choose n random points
    xyz_len = source_xyz.shape[1]
    if not xyz_len >= n_keypoints:
        raise Exception(
            "Number of keypoints {} exceeds the number of extracted features {}".format(n_keypoints, xyz_len))

    indexes = np.random.choice(xyz_len, n_keypoints, replace=False)
    source_xyz = source_xyz[:, indexes]
    source_features = source_features[indexes, :]

    corrs_s, corrs_t = correspondence_helpers.find_correspondences(
        source_features, target_features, distance_metric='euclidean', mutual_filter=False)

    logging.debug("Number of target features: " + str(len(target_features)))
    logging.debug("Number of source features: " + str(len(source_features)))
    logging.debug("Number of correspondences: " + str(len(corrs_s)))

    registration_solution = np.eye(4)
    if algorithm == 'TEASER':
        registration_solution = teaser.run_teaser_registration(config, source_xyz, target_xyz, corrs_s,
                                                               corrs_t)
    elif algorithm == 'RANSAC':
        registration_solution = ransac.run_ransac_registration(config, source_xyz, target_xyz, corrs_s,
                                                             corrs_t)
    return registration_solution


if __name__ == '__main__':
    source_npz_test = "/tmp/source.npz"
    target_npz_test = "/tmp/target.npz"
    registration_result = register(source_npz_test, target_npz_test)