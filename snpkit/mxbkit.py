from . import io
from tqdm import tqdm
import adelie as ad
import adelie.logger as logger
import numpy as np
import pandas as pd
import snpkit as sk


def _msp_get_fid(
    sample_IDs: list,
):
    # sample ID is of the form FID_IID
    return np.array([int(x.split("_")[0]) for x in sample_IDs])


def _msp_get_iid(
    sample_IDs: list,
):
    # sample ID is of the form FID_IID
    return np.array([int(x.split("_")[1]) for x in sample_IDs])


def read_fam(
    fam: str,
    clean: bool =True,
):
    fam_df = pd.read_csv(
        fam, 
        sep=" ", 
        header=None, 
        names=['FID', 'IID', 'father', 'mother', 'gender', 'trait'],
    )

    if clean:
        # so stupid that there are duplicated rows
        fam_df.set_index("IID", inplace=True)
        to_drop_indices = np.arange(fam_df.shape[0])[fam_df.index.duplicated()]
        fam_df.reset_index(inplace=True)
        fam_df.drop(to_drop_indices, inplace=True)
        fam_df.reset_index(inplace=True, names="fam_index")

        fam_df["FID"] = fam_df["FID"].astype(int)

        assert fam_df["FID"].shape[0] == fam_df["FID"].unique().shape[0]
        assert fam_df["IID"].shape[0] == fam_df["IID"].unique().shape[0]

    return fam_df


def read_bim(
    bim: str,
):
    bim_df = pd.read_csv(
        bim, 
        sep="\t", 
        header=None, 
        names=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1'],
    )
    return bim_df


def check_msp_coherence(
    msps: list,
):
    sample_IDs_list = []
    for msp in msps:
        reader = io.MSPReader(msp)
        reader.read_header()
        sample_IDs_list.append(reader.sample_IDs)

    # check that every MSP file has same sample_IDs
    for i in range(len(sample_IDs_list)):
        assert sample_IDs_list[i] == sample_IDs_list[0]

    # check that the FIDs are sorted 
    fids = _msp_get_fid(sample_IDs_list[0])
    assert np.all(fids == np.sort(fids))

    # check that FIDs are unique
    assert fids.shape[0] == np.unique(fids).shape[0]


def check_call_coherence(
    fams: list,
):
    iids = read_fam(fams[0])["IID"]
    for fam in fams:
        assert list(iids) == list(read_fam(fam)["IID"])


def check_call_msp_coherence(
    fam: str,
    msp: str,
):
    fam_df = read_fam(fam)
    reader = io.MSPReader(msp)
    reader.read_header()

    # check MSP FIDs are in FAM FIDS
    fids = _msp_get_fid(reader.sample_IDs)
    fam_fids = np.array(fam_df["FID"])
    for fid in fids:
        assert fid in fam_fids, f"{fid} does not exist in FAM!"


def read_calldata(
    bed: str,
    fam: str,
    *,
    sample_indices: np.ndarray =None,
    snp_indices: np.ndarray =None,
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

    # get raw_sample_ct
    fam_df = read_fam(fam, clean=False)
    raw_sample_ct = fam_df.shape[0]

    # instantiate PGEN reader
    pgen_reader = pg.PgenReader(
        str.encode(bed),
        raw_sample_ct=raw_sample_ct,
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
    calldata = sk.to_sample_major(calldata, n_threads=n_threads)

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
    lai = sk.to_sample_major(lai, n_threads=n_threads)

    return lai, sample_indices, reader


def cache_unphased_snpdat(
    bed: str,
    fam: str,
    dest: str,
    *,
    fam_indices: np.ndarray=None,
    msp_indices: np.ndarray=None,
    snp_indices: np.ndarray=None,
    sort_indices: bool =True,
    n_threads: int =1,
):
    logger.logger.info(f"Reading PGEN metadata files.")

    (
        calldata,
        _,
        _,
    ) = read_calldata(
        bed,
        fam,
        sample_indices=fam_indices,
        snp_indices=snp_indices,
        sort_indices=False,
        n_threads=n_threads,
    )

    calldata = sk.calldata_sum(calldata, n_threads=n_threads)

    # convert to snpdat
    logger.logger.info("Saving as snpdat.")
    handler = ad.io.snp_unphased(dest)
    bytes_written = handler.write(
        calldata=calldata,
        n_threads=n_threads,
    )

    logger.logger.info(f"Bytes written: {bytes_written}")


def cache_snpdat(
    bed: str,
    bim: str,
    fam: str,
    msp: str,
    dest: str,
    *,
    fam_indices: np.ndarray=None,
    msp_indices: np.ndarray=None,
    snp_indices: np.ndarray=None,
    n_threads: int =1,
):
    logger.logger.info(f"Reading PGEN metadata files.")
    bim_df = read_bim(bim)
    fam_df = read_fam(fam)

    (
        lai,
        _,
        reader,
    ) = read_lai(
        msp,
        bim_df["pos"][snp_indices],
        sample_indices=msp_indices,
        sort_indices=False,
        n_threads=n_threads,
    )

    (
        calldata,
        _,
        _,
    ) = read_calldata(
        bed,
        fam,
        sample_indices=fam_indices,
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