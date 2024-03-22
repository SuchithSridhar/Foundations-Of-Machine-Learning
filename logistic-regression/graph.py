import matplotlib.pyplot as plt


def plot_clusters(clusters):
    plt.figure(figsize=(10, 6))
    unique_ids = set(clusters[:, 2])
    for id in unique_ids:
        id_points = clusters[clusters[:, 2] == id]
        plt.scatter(id_points[:, 0], id_points[:, 1], label=f"Group-{int(id)}")
    plt.legend()
    plt.show()


def plot_train_val(train, val):
    unique_ids = set(train[:, 2])
    plt.figure(figsize=(10, 6))
    for id in unique_ids:
        id_points_train = train[train[:, 2] == id]
        plt.scatter(
            id_points_train[:, 0],
            id_points_train[:, 1],
            label=f"Train Group-{int(id)}"
        )

        if val is not None and len(val) > 0:
            id_points_val = val[val[:, 2] == id]
            plt.scatter(id_points_val[:, 0], id_points_val[:, 1],
                        marker='*', label=f"Val Group-{int(id)}")

    plt.title("Plot for predictions.")
    plt.legend()
    plt.show()
