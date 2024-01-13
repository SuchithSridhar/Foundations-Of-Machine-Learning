#!/bin/python3
# Author: Suchith Sridhar
# Date: 2024, Jan 13
# Minimum python version: 3.11.x
# My solution to classifying points using
# k-nearest-neighbors

import numpy as np


def distance(p1: np.array, p2: np.array) -> np.double:
    '''
    Find the Euclidean distance between two points.

    Parameters:
        p1 (np.array): An n dimensional point.
        p2 (np.array): An n dimensional point.

    Returns:
        The Euclidean distance between points p1 and p2.
    '''

    dist = np.linalg.norm(p1 - p2)
    return dist


def classify_point(p: np.array, dataset: np.array,
                   classifcations: np.array, neighbors: int) -> int:
    '''
    Classify point using k-nearest-neighbors method of classification.

    Parameters:
        p (np.array): The point to be classified.
        dataset (np.array): The set of points that have already
            been classified.
        classifications (np.array): the corresponding classification
            for dataset.
        neighbors (int): The number of neighbors to compare against.

    Returns:
        The classification of the point p based on the k-nearest-neighbors
        method of classification.
    '''

    distances = []
    for i, point in enumerate(dataset):
        # We're calculating the distance and storing the index along with it.
        distances.append((distance(p, point), i))

    # We sort by the distance
    distances.sort(key=lambda x: x[0])

    closest_k = distances[0:neighbors]

    votes = {}

    for i, point in enumerate(closest_k):
        classy = classifcations[point[1]]
        if classy not in votes:
            votes[classy] = 1
        else:
            votes[classy] += 1

    # find the maximum votes
    max = -1
    max_key = 0
    for key, value in votes.items():
        if value > max:
            max = value
            max_key = key

    return max_key


def classify_points(points: np.array, dataset: np.array,
                    classifcations: np.array, neighbors: int) -> int:
    '''
    Classify an array of points using k-nearest-neighbors method of
    classification.

    Parameters:
        points (np.array): An array of n dimensional points
        dataset (np.array): The set of points that have already
            been classified.
        classifications (np.array): the corresponding classification
            for dataset.
        neighbors (int): The number of neighbors to compare against.

    Returns:
        An array of classifications for each of the points in points.
    '''

    pt_class = np.zeros(points.shape[0], dtype=int)
    for i, tp in enumerate(points):
        pt_class[i] = classify_point(tp, dataset, classifcations, neighbors)

    return pt_class
