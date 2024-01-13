#!/bin/python3
# Author: Suchith Sridhar
# Date: 2024, Jan 13
# Minimum python version: 3.11.x
# Utils to generate and display data.

import numpy as np
import matplotlib.pyplot as plt


def generate_points(dimensions: int, count: int, noise: int,
                    classification_groups: int) -> (np.array, np.array):
    '''
    Generate points based on the given parameters. The generated points are
    classified into a group from 0 to the number of groups.

    Parameters:
        dimensions (int): The number of dimensions in which points should be
            generated.
        count (int): The number of points to be generated.
        noise (int): The amount of noise that should be inserted into the
            generated points. This must be a value from 1 to 100.
        classification_groups (int): The number of groups the points should be
            classified into. Ensure this value is greater than the number of
            points.

    Returns:
        A tuple where the first value is the generated points and the second
        value is the classification of each of those points.
    '''

    if classification_groups <= 1 or classification_groups > count:
        raise ValueError("classification_groups must be greater than 1 and"
                         " less than or equal to count")
    if not (1 <= noise <= 100):
        raise ValueError("noise must be between 1 and 100")

    centroids = np.random.rand(classification_groups, dimensions)

    points = np.zeros((count, dimensions))
    classifications = np.zeros(count, dtype=int)

    for i in range(count):
        # Choose a random classification group
        group = np.random.randint(0, classification_groups)
        classifications[i] = group

        # Generate point near the centroid of the chosen group
        base_point = centroids[group, :]
        noise_factor = (noise/100) * np.random.randn(dimensions)
        points[i, :] = base_point + noise_factor

    return points, classifications


def generate_test_points(dimensions: int, count: int) -> np.array:
    '''
    Generate random points in the space to be used for classification.

    Parameters:
        dimensions (int): The dimensions of the space the points are to be
            generated.
        count (int): The number of points to be generated.

    Return:
        A numpy array of points, each of which is an $dimensions
        sized numpy array.
    '''
    return np.random.rand(count, dimensions)


def plot_2d_points(points: np.array, classifications: np.array):
    '''
    Plots the generated 2D points in a scatter plot, color-coded by their
            classification.

    Parameters:
        points (np.array): The array of 2D points generated.
        classifications (np.array): Array of classifications for each point.
    '''
    if points.shape[1] != 2:
        raise ValueError("Points array must have exactly 2"
                         " dimensions for each point.")

    unique_classes = np.unique(classifications)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

    for group, color in zip(unique_classes, colors):
        group_points = points[classifications == group]
        plt.scatter(group_points[:, 0], group_points[:, 1],
                    color=color, label=f'Group {group}')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Scatter Plot of Generated Points')
    plt.legend()
    plt.show()


def plot_2d_test_against_data(points: np.array, classifications: np.array,
                              test_points: np.array,
                              test_classifications: np.array):

    if points.shape[1] != 2 and test_points.shape[1] != 2:
        raise ValueError("Points array must have exactly 2"
                         " dimensions for each point.")

    unique_classes = np.unique(classifications)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

    for group, color in zip(unique_classes, colors):
        group_points = points[classifications == group]
        group_test_points = test_points[test_classifications == group]
        plt.scatter(group_points[:, 0], group_points[:, 1],
                    color=color, label=f'Group {group}', alpha=0.2)
        plt.scatter(group_test_points[:, 0], group_test_points[:, 1],
                    color=color, marker="*", label=f'Test Group {group}')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Scatter of test points against data')
    plt.legend()
    plt.show()
