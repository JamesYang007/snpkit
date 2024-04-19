import snpkit as sk
import snpkit.snpkit_core as core
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


def test_column_mean():
    n = 217
    p = 113
    n_threads = 8
    seed = 0
    np.random.seed(seed)
    mat = np.asfortranarray(np.random.binomial(1, 0.5, (n, 2 * p)).astype(np.int8))
    nans = np.random.choice(n*p, size=(n*p)//2, replace=False)
    nans = np.unravel_index(nans, shape=(n,p))
    mat[nans] = -9
    actual = core.column_mean(mat, n_threads)
    expected1 = np.nanmean(mat, 0, where=(mat>=0))
    expected2 = (mat == -9).sum(0)
    assert np.allclose(actual[0], expected1)
    assert np.allclose(actual[1], expected2)