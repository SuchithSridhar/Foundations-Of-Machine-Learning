from logistic_regression import LogisticRegression
import graph
import numpy as np


def generate_cluster(mean: tuple[float, float], cov: list[list[float, float]],
                     size: int, id: int):

    # standard normal distribution for both variables.
    array = np.random.multivariate_normal(mean, cov, size)
    new_column = np.full((array.shape[0], 1), id)
    result = np.concatenate((array, new_column), axis=1)
    return result


def main():
    SIZE = 100
    means = [[10, 0], [0, 6], [3, 1]]
    cov = [[1, 0], [0, 1]]
    clusters = [
        generate_cluster(mean, cov, SIZE, id) for id, mean in enumerate(means)
    ]
    clusters = np.concatenate(clusters)

    # Shuffle cluster to mix-up data points
    np.random.shuffle(clusters)
    split_index = int(0.8 * clusters.shape[0])

    train = clusters[:split_index]
    val_w_labels = clusters[split_index:]

    LABEL_COLUMN = 2

    # Remove all labels before trying to classify
    val = val_w_labels.copy()
    val[:, LABEL_COLUMN] = -1

    model = LogisticRegression()
    model.fit(train[:, 0:-1], train[:, LABEL_COLUMN])
    predictions = model.predict(val[:, 0:-1])
    val[:, LABEL_COLUMN] = predictions

    graph.plot_train_val(train, val)


if __name__ == "__main__":
    main()
