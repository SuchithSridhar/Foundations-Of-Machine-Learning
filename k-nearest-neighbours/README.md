# K Nearest Neighbors Implementation

A simple and most likely inefficient implementation of k-nearest-neighbors.
There are 3 files in this code base:

- `utils.py`: Contains utility functions to generate points and plot points.
- `soultion.py`: Contains the classifier function and the solution to the
  problem.
- `main.py`: The main sequence of the program.


**How the random data is generated**:

The data generated is based on generating random centroid points and then
generating points around this centroid base and then maybe adding some noise. So
our "random points" are going to be in circular regions. This was designed this
way to facilitate easy classification.

## Try it yourself

If you'd like to try writing this on your own, just make sure you just use
`utils.py` and not `solution.py`. All the function in `utils.py` have been
documented.

Note:

- `points`: is an C x N array where each point is `N` dimensional.
- `classification`: is a C X 1 array where each item is the classification of
  the point at the corresponding index.

So, the classification of `points[i]` is `classification[i]`.

Generate points using:

```python
pts, pts_class = utils.generate_points(dims, data_count, noise, groups)
```

Graph the generated points using:

```python
utils.plot_2d_points(pts, pts_class)
```

Generate test points using:

```python
test_pts = utils.generate_test_points(dims, test_count)
```

Once you've classified the points, you can graph them against the generated
points to see the results using:

```python
utils.plot_2d_test_against_data(pts, pts_class, test_pts, test_pts_class)
```
