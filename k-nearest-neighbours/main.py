#!/bin/python3
# Author: Suchith Sridhar
# Date: 2024, Jan 12
# Minimum python version: 3.11.x

import utils
import solution

dims = 2
data_count = 200
noise = 12
groups = 2
test_count = 20

neighbors = 5

pts, pts_class = utils.generate_points(dims, data_count, noise, groups)
test_pts = utils.generate_test_points(dims, test_count)

# Use this to view generated points
# utils.plot_2d_points(pts, pts_class)

# Classify points using k-nearest-neighbors
test_pts_class = solution.classify_points(test_pts, pts, pts_class, neighbors)

utils.plot_2d_test_against_data(pts, pts_class, test_pts, test_pts_class)
