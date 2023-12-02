import logging
import adelie as ad
import adelie.logger as logger
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm


def phe_to_csv(
    src: str,
    phenotype: str,
    dest: str,
):
    """Converts master phenotype file to a smaller CSV file with one phenotype.

    Parameters
    ----------
    src : str
        Master phenotype filename.
    phenotype : str
        Phenotype ID in master phenotype file.
    dest : str
        Smaller CSV filename.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with IID and given phenotype.
    """
    logger.logger.info("Reading master CSV.")
    master_df = pd.read_csv(src, sep='\t', usecols=['IID', phenotype])
    logger.logger.info(f"Saving IID/{phenotype} CSV.")
    master_df.to_csv(dest, sep=",", index=False)
    return master_df


def pickle_msp(
    src: str, 
    dest: str,
    iids: np.ndarray,
):
    """Reads and pickles MSP object.

    TODO: is this necessary?

    Parameters
    ----------
    src : str
        MSP filename.
    dest : str
        Pickle filename.
    iids : np.ndarray
        Subset of IIDs.
    
    Returns
    -------
    snpobj
        The resulting object from reading the MSP file using ``msp_reader.MSPReader``.
    """
    from . import msp_reader
    reader = msp_reader.MSPReader(src)
    logger.logger.info(f"Reading MSP file.")
    snpobj = reader.read(
        usecols=reader.iid_to_msp_cols(iids),
    )
    logger.logger.info("Pickling MSP object.")
    with open(dest, 'wb') as handle:
        pickle.dump(snpobj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return snpobj


def gen_to_snpdat(
    pgen: str,    
    pvar: str,
    psam: str,
    msp: str,
    dest: str,
    *,
    maf_tol: float=0.01,
    n_threads: int =1,
):
    """Converts a PGEN file to CSV file.

    Parameters
    ----------
    src : str
        PGEN filename.
    dest : str
        CSV filename.
    """
    # pgenlib is a really poorly written library that doesn't work on Mac M1.
    # We do not perform a global import since this module cannot be loaded on Mac M1 then.
    import pgenlib as pg

    # instantiate PGEN reader
    pgen_reader = pg.PgenReader(
        str.encode(pgen),
    )

    # create calldata array
    calldatas = []
    calldata_shape = (2 * pgen_reader.get_raw_sample_ct(),)
    calldata = np.empty(
        calldata_shape,
        dtype=np.int32,
    )
    active_snps = []

    logger.logger.info(f"Reading PGEN file.")
    for i in tqdm(range(pgen_reader.get_variant_ct())): 
        pgen_reader.read_alleles(i, calldata)
        if np.mean(calldata, dtype=float) >= maf_tol:
            calldatas.append(calldata.astype(np.int8))
            active_snps.append(i)
    pgen_reader.close()

    # store calldata
    calldata = np.vstack(calldatas, dtype=np.int8)
    calldata = np.transpose(
        calldata.reshape((calldata.shape[0], calldata.shape[1] // 2, 2)),
        (1, 0, 2),
    )
    calldata = calldata.reshape((calldata.shape[0], -1))
    calldata = np.array(calldata, copy=False, order="F", dtype=np.int8)

    # load PGEN metadata
    logger.logger.info(f"Loading PGEN metadata.")
    pvar_df = pd.read_csv(
        pvar, 
        sep='\t', 
        comment='#',
        header=None, 
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT'],
        dtype={'CHROM': str},
    )
    psam_df = pd.read_csv(psam, sep='\t')

    # instantiate MSP object
    logger.logger.info(f"Loading MSP object.")
    from . import msp_reader
    reader = msp_reader.MSPReader(msp)
    snpobj = reader.read(n_threads=n_threads)
    # the non-negative IIDs (not excluded by UKB) should match up exactly.
    assert np.allclose(
        snpobj.sample_IDs[snpobj.sample_IDs >= 0], 
        psam_df["IID"][psam_df["IID"] >= 0],
    )

    logger.logger.info("Expanding LAI.")
    lai_indices = np.searchsorted(
        snpobj.physical_pos[:, 0], 
        pvar_df['POS'][active_snps], 
        side='right',
    ) - 1
    assert np.all(lai_indices >= 0)
    lai = snpobj.lai[lai_indices]
    lai = np.transpose(
        lai.reshape((lai.shape[0], lai.shape[1] // 2, 2)),
        (1, 0, 2),
    )
    lai = lai.reshape((lai.shape[0], -1))
    lai = np.array(lai, copy=False, order="C", dtype=np.int8)

    assert calldata.shape == lai.shape

    # convert to snpdat
    logger.logger.info("Saving as snpdat.")
    logger.logger.info(f"Number of ancestries {snpobj.n_ancestries}")
    handler = ad.io.snp_phased_ancestry(dest)
    bytes_written = handler.write(
        calldata=calldata,
        ancestries=lai,
        A=snpobj.n_ancestries,
        n_threads=n_threads,
    )

    return active_snps, calldata, lai, bytes_written


def common_iids(
    phe: str,
    msp: str,
    psam: str, 
):
    # get non-missing IIDs in phenotype
    master_df = pd.read_csv(phe)
    phe_iids = (master_df.iloc[:, 1] != -9).to_numpy()
    phe_iids = master_df.iloc[phe_iids, 0]

    # get MSP IIDs and intersect with previous
    from . import msp_reader
    reader = msp_reader.MSPReader(msp)
    header = reader.read_header()
    msp_iids = pd.Series(reader.sample_IDs())
    msp_iids = msp_iids[msp_iids.isin(phe_iids)]

    # get calldata IIDs and intersect with previous
    samples_info = pd.read_csv(psam, sep='\t')
    psam_iids = samples_info["IID"]
    psam_iids = samples_info.loc[psam_iids.isin(msp_iids).to_numpy(), 'IID']

    common_iids = psam_iids

    return (
        phe_iids[phe_iids.isin(common_iids)].index.to_numpy(dtype=np.uint32), 
        msp_iids[msp_iids.isin(common_iids)].index.to_numpy(dtype=np.uint32), 
        psam_iids.index.to_numpy(dtype=np.uint32),
        psam_iids.to_numpy(dtype=np.uint32),
    )
