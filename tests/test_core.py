import snpkit as sk
import numpy as np


def test_to_sample_major():
    n = 217
    p = 113
    n_threads = 8
    seed = 0
    np.random.seed(seed)
    mat = np.random.binomial(1, 0.5, (p, 2 * n)).astype(np.int8)
    actual = sk.to_sample_major(mat, n_threads=n_threads)
    expected = np.transpose(mat.reshape((p, n, 2)), (1, 0, 2)).reshape(n, 2 * p)
    assert np.allclose(actual, expected)


def test_calldata_sum():
    n = 217
    p = 113
    n_threads = 8
    seed = 0
    np.random.seed(seed)
    mat = np.asfortranarray(np.random.binomial(1, 0.5, (n, 2 * p)).astype(np.int8))
    actual = sk.calldata_sum(mat, n_threads=n_threads)
    expected = np.sum(mat.reshape((n, p, 2)), axis=-1)
    assert np.allclose(actual, expected)


def test_subset_rows_cols():
    n = 217
    p = 113
    n_threads = 8
    seed = 0
    np.random.seed(seed)
    mat = np.asfortranarray(np.random.binomial(1, 0.5, (n, 2 * p)).astype(np.int8))
    row_indices = np.random.randint(0, n, n // 2)
    col_indices = np.random.randint(0, p, p // 2)
    actual = sk.calldata_subset_rows_cols(mat, row_indices, col_indices, n_threads)
    expected = mat[row_indices].reshape((len(row_indices), p, 2))[:, col_indices, :].reshape((len(row_indices), -1))
    assert np.allclose(actual, expected)
