"""The ten training digits of the FEP-ANN digits task."""


def load_digits(n_train=10):
    """FEP-ANN preprocessing: squared pixel values, each image z-scored; first n_train images for training."""
    from sklearn import datasets
    d = datasets.load_digits().data.astype(float) ** 2
    d = (d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)
    return d[:n_train], d[n_train:]

