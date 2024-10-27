from . import snpkit_core as core
import numpy as np


def to_sample_major(
    mat: np.ndarray,
    n_threads: int =1,
):
    """Converts SNP-major to Sample-major data.

    Sample-major data is of the form ``X[i,j,k]`` with
    individual ``i``, SNP ``j``, and haplotype ``k``.
    Then, SNP-major data is of the form ``Y[j,i,k]``.

    Parameters
    ----------
    mat : (s, 2 * n) np.ndarray
        SNP-major array.

    Returns
    -------
    out : (n, 2 * s) np.ndarray
        Sample-major array.
    """
    dispatcher = {
        np.dtype("int8"): core.to_sample_major_int8,
    }
    return dispatcher[mat.dtype](mat, n_threads)


def calldata_sum(
    calldata: np.ndarray,
    n_threads: int =1,
):
    dispatcher = {
        np.dtype("int8"): core.calldata_sum_int8,
    }
    return dispatcher[calldata.dtype](calldata, n_threads)


def calldata_subset_rows_cols(
    mat: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    n_threads: int =1,
):
    dispatcher = {
        np.dtype("int8"): core.calldata_subset_rows_cols_int8,
    }
    return dispatcher[mat.dtype](mat, row_indices, col_indices, n_threads)


def maf_subset(
    calldata: np.ndarray,
    maf_tol: float =0.01,
    *,
    n_threads: int =1,
):
    mafs = 0.5 * core.column_mean(calldata, n_threads)[0]
    return np.arange(mafs.shape[-1])[mafs >= maf_tol]
