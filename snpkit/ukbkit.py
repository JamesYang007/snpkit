import adelie as ad
import adelie.logger as logger
import pandas as pd
import numpy as np
import pickle
import snpkit as sk
import snpkit.snpkit_core as core
import snpkit.io
import warnings
from tqdm import tqdm
from typing import Union


def permute_by(
    x: pd.Series,
    y: pd.Series,
):
    """Permute a Series so that it is equivalent to another Series entry-wise.
    """
    def _transform(z):
        z = z.astype(int)
        z = z.reset_index()
        z = z.set_index(z.iloc[:, 1])
        z = z.drop(columns=[z.columns[-1]])
        return z
    x = _transform(x)
    y = _transform(y)

    out = x.loc[y.index]
    out = out.reset_index()
    out = out.set_index(out.iloc[:, 1])
    out = out.drop(columns=[out.columns[-1]])
    out = out.iloc[:, 0]
    out.index.name = None
    return out


def check_call_coherence(
    psams: list,
):
    iids = pd.read_csv(psams[0], sep="\t")["IID"]
    for psam in psams:
        assert np.allclose(iids, pd.read_csv(psam, sep="\t")["IID"])


def check_msp_coherence(
    msps: list,
):
    reader = sk.io.MSPReader(msps[0])
    reader.read_header()
    iids = reader.sample_IDs
    for msp in msps:
        reader = sk.io.MSPReader(msp)
        reader.read_header()
        assert np.allclose(iids, reader.sample_IDs)


def check_call_msp_coherence(
    psam: str, 
    msp: str,
    indices: np.ndarray =None,
):
    psam_df = pd.read_csv(psam, sep='\t')
    reader = sk.io.MSPReader(msp)
    reader.read_header()
    psam_iids = psam_df["IID"].to_numpy()
    msp_iids = reader.sample_IDs

    if indices is None:
        if len(psam_iids) != len(msp_iids):
            warnings.warn(f"IIDs have different lengths:\nPSAM: {psam}\nMSP:  {msp}")
        bad_idxs = psam_iids != msp_iids
        if np.any(psam_iids[bad_idxs] >= 0) or np.any(msp_iids[bad_idxs] >= 0):
            warnings.warn(f"IIDs that do not match exactly have some non-negative ID:\nPSAM: {psam}\nMSP:  {msp}")
    else:
        if np.any(psam_iids[indices] != msp_iids[indices]):
            warnings.warn("IIDs do not match exactly on the given indices!")
    return psam_iids, msp_iids


def read_calldata(
    pgen: str,
    *,
    sample_indices: np.ndarray=None,
    snp_indices: np.ndarray=None,
    sort_indices: bool =True,
    n_threads: int =1,
):
    if sort_indices:
        if not sample_indices is None:
            sample_indices = np.sort(sample_indices)
        if not snp_indices is None:
            snp_indices = np.sort(snp_indices)

    # pgenlib is a really poorly written library that doesn't work on Mac M1.
    # We do not perform a global import since this module cannot be loaded on Mac M1 then.
    import pgenlib as pg

    # instantiate PGEN reader
    pgen_reader = pg.PgenReader(
        str.encode(pgen),
        sample_subset=sample_indices,
    )

    if sample_indices is None:
        sample_indices = np.arange(pgen_reader.get_raw_sample_ct())
    if snp_indices is None:
        snp_indices = np.arange(pgen_reader.get_variant_ct())

    # create calldata array
    calldata = np.empty(
        (len(snp_indices), 2 * len(sample_indices)),
        dtype=np.int8,
    )
    calldata_row = np.empty(
        2 * len(sample_indices),
        dtype=np.int32,
    )

    # get active SNPs among the current subset with MAF >= threshold
    # and the corresponding rows of calldata.
    logger.logger.info(f"Reading PGEN file.")
    for i in tqdm(range(len(snp_indices))): 
        pgen_reader.read_alleles(snp_indices[i], calldata_row)
        calldata[i] = calldata_row
    pgen_reader.close()

    logger.logger.info(f"Converting to sample-major.")
    calldata = to_sample_major(calldata, n_threads=n_threads)

    return calldata, sample_indices, snp_indices


def read_lai(
    msp: str,
    snps: np.ndarray,
    *,
    sample_indices: np.ndarray=None,
    sort_indices: bool =True,
    n_threads: int =1,
):
    if sort_indices:
        if not sample_indices is None:
            sample_indices = np.sort(sample_indices)

    reader = sk.io.MSPReader(msp)

    if sample_indices is None:
        reader.read_header()
        sample_indices = np.arange(len(reader.sample_IDs))

    hap_ids_indices = np.empty(2 * len(sample_indices), dtype=np.uint32)
    hap_ids_indices[::2] = 2 * sample_indices
    hap_ids_indices[1::2] = hap_ids_indices[::2] + 1

    logger.logger.info(f"Reading MSP file.")
    reader.read(
        hap_ids_indices=hap_ids_indices,
        n_threads=n_threads,
    )

    logger.logger.info(f"Searching for LAI indices for each SNP position.")
    lai_indices = np.searchsorted(
        reader.pos[:, 0], 
        snps, 
        side='right',
    ) - 1
    assert np.all(lai_indices >= 0), "Calldata SNP position not found in the expected range for LAI!"

    logger.logger.info(f"Expanding LAI per SNP.")
    lai = reader.lai[lai_indices]

    logger.logger.info(f"Converting to sample-major.")
    lai = to_sample_major(lai, n_threads=n_threads)

    return lai, sample_indices, reader


def cache_snpdat(
    pgen: str,
    pvar: str,
    psam: str,
    msp: str,
    dest: str,
    *,
    sample_indices: np.ndarray=None,
    snp_indices: np.ndarray=None,
    sort_indices: bool =True,
    n_threads: int =1,
):
    logger.logger.info(f"Reading PGEN metadata files.")
    pvar_df = pd.read_csv(
        pvar, 
        sep='\t', 
        comment='#',
        header=None, 
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT'],
        dtype={'CHROM': str},
    )
    psam_df = pd.read_csv(psam, sep='\t')

    (
        lai,
        sample_indices,
        reader,
    ) = read_lai(
        msp,
        pvar_df["POS"][snp_indices],
        sample_indices=sample_indices,
        sort_indices=sort_indices,
        n_threads=n_threads,
    )

    # the non-negative IIDs (not excluded by UKB) should match up exactly.
    psam_iids_subset = psam_df.iloc[sample_indices, :]["IID"]
    assert np.allclose(
        reader.sample_IDs[reader.sample_IDs >= 0], 
        psam_iids_subset[psam_iids_subset >= 0],
    )

    (
        calldata,
        _,
        _,
    ) = read_calldata(
        pgen,
        sample_indices=sample_indices,
        snp_indices=snp_indices,
        sort_indices=False,
        n_threads=n_threads,
    )

    assert calldata.shape == lai.shape, "Calldata and LAI do not have the same shape!"

    # convert to snpdat
    logger.logger.info("Saving as snpdat.")
    n_ancestries = len(reader.ancestry_map)
    logger.logger.info(f"Number of ancestries {n_ancestries}")
    handler = ad.io.snp_phased_ancestry(dest)
    bytes_written = handler.write(
        calldata=calldata,
        ancestries=lai,
        A=n_ancestries,
        n_threads=n_threads,
    )

    logger.logger.info(f"Bytes written: {bytes_written}")


def cache_unphased_snpdat(
    pgen: str,
    pvar: str,
    psam: str,
    dest: str,
    *,
    sample_indices: np.ndarray=None,
    snp_indices: np.ndarray=None,
    sort_indices: bool =True,
    n_threads: int =1,
):
    logger.logger.info(f"Reading PGEN metadata files.")
    pvar_df = pd.read_csv(
        pvar, 
        sep='\t', 
        comment='#',
        header=None, 
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT'],
        dtype={'CHROM': str},
    )
    psam_df = pd.read_csv(psam, sep='\t')

    (
        calldata,
        _,
        _,
    ) = read_calldata(
        pgen,
        sample_indices=sample_indices,
        snp_indices=snp_indices,
        sort_indices=False,
        n_threads=n_threads,
    )

    calldata = calldata_sum(calldata, n_threads=n_threads)

    # convert to snpdat
    logger.logger.info("Saving as snpdat.")
    handler = ad.io.snp_unphased(dest)
    bytes_written = handler.write(
        calldata=calldata,
        n_threads=n_threads,
    )

    logger.logger.info(f"Bytes written: {bytes_written}")


def valid_iids(
    phe: str,
    msp: Union[str, list],
    psam: Union[str, list], 
):
    if (
        isinstance(msp, list) and
        isinstance(psam, list)
    ):
        phe_iids = None
        common_iids = None
        for msp_i, psam_i in zip(msp, psam):
            (
                next_phe_iids,
                next_common_iids,
            ) = valid_iids(
                phe=phe,
                msp=msp_i,
                psam=psam_i,
            )
            if phe_iids is None and common_iids is None:
                phe_iids = next_phe_iids
                common_iids = next_common_iids
                continue
            phe_iids = phe_iids[phe_iids.isin(next_phe_iids)]
            common_iids = common_iids[common_iids.isin(next_common_iids)]
        return phe_iids, common_iids
    elif (
        isinstance(msp, str) and
        isinstance(psam, str)
    ):
        # get non-missing phenotype data
        master_df = pd.read_csv(phe)
        phe_iids = master_df["IID"]

        # get MSP IIDs with non-missing phenotype
        reader = sk.io.msp_reader.MSPReader(msp)
        header = reader.read_header()
        msp_iids = pd.Series(reader.sample_IDs)
        common_iids = msp_iids[msp_iids.isin(phe_iids)]
        logger.logger.info(f"Number of discarded MSP IIDs: {msp_iids.shape[0] - common_iids.shape[0]}")

        # get calldata IIDs and intersect with previous
        samples_info = pd.read_csv(psam, sep='\t')
        psam_iids = samples_info["IID"]
        assert np.sum(msp_iids[msp_iids >= 0] != psam_iids[psam_iids >= 0]) == 0, \
            "MSP and PSAM should have same IIDs along non-negative IIDs!"

        return (
            phe_iids[phe_iids.isin(common_iids)], 
            common_iids,
        )
    raise RuntimeError("Unexpected input types. Must be all lists or all strings.")
