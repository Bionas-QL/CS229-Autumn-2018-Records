from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def K_means(A, iterations=30, num_cluster=16):
    """
    Args:
        A: The image Numpy array of shape (m, n, 3)
        iterations: The smallest amount of iterations
        num_cluster : The number of clusters we would like to obtain

    Returns:
        centroids: Numpy array of shape (num_cluster, 1, 1, 3)
    """

    def update_centroids(A, centroids):
        """
        Args:
            A: The image Numpy array of shape (m, n, 3)
            centroids: Numpy array of shape (num_cluster, 1, 1, 3)
        """
        # The array of centroids is updated in place
        closest_centroid = np.argmin(np.linalg.norm(A - centroids, axis=3), axis=0)
        for i in range(num_cluster):
            cluster = A[closest_centroid == i]
            if cluster.shape[0]:
                centroids[i, 0, 0] = np.mean(cluster, axis=0)

    m, n, _ = A.shape
    # Initialize centroids. Note that it is important to make it of type float64 to avoid roundings to integers.
    # We reshape the array to make full use of broadcasting.
    centroids = A[np.random.randint(m, size=num_cluster), np.random.randint(n, size=num_cluster)].reshape(
        (num_cluster, 1, 1, 3)).astype("float64")

    for _ in range(iterations):
        old_centroids = np.copy(centroids)
        update_centroids(A, centroids)
        if (centroids == old_centroids).all():
            break
        centroids = old_centroids

    return centroids


def main():
    A = imread('../data/peppers-large.tiff')
    smallImage = imread('../data/peppers-small.tiff')
    m, n, _ = A.shape
    newImage = np.zeros((m, n, 3))
    centroids = K_means(smallImage)
    closest_centroid = np.argmin(np.linalg.norm(A - centroids, axis=3), axis=0)
    for i in range(m):
        for j in range(n):
            newImage[i, j] = centroids[closest_centroid[i, j], 0, 0]
    newImage = newImage.astype(int)
    plt.imshow(newImage)
    plt.show()


if __name__ == "__main__":
    main()
