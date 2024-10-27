from .. import snpkit_core as core
import ctypes
import logging
import numpy as np
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)


class MSPReader(core.io.MSPReader):
    def __init__(self, filename):
        self.base_type = core.io.MSPReader
        self.base_type.__init__(self, filename)
        self._filename = filename

    def read_header(
        self,
        *,
        delimiter="\t",
        buffer_size=1 << 22,
        n_rows_hint=2000,
        n_threads=1, 
    ):
        self.base_type.read(
            self,
            max_rows=2,
            delimiter=delimiter,
            hap_ids_indices=[],
            buffer_size=buffer_size,
            n_rows_hint=n_rows_hint,
            n_threads=n_threads,
        )

    def read(
        self, 
        *, 
        max_rows=None,
        delimiter="\t",
        hap_ids_indices=None,
        buffer_size=1 << 22,
        n_rows_hint=2000,
        n_threads=1, 
        debug=False,
    ):
        log.info(f"Reading msp file: {self.filename}")

        if max_rows is None:
            max_rows = ctypes.c_uint64(-1).value

        # if user provides an explicit list, it must be non-empty
        if isinstance(hap_ids_indices, list) or isinstance(hap_ids_indices, np.ndarray):
            assert len(hap_ids_indices), "hap_ids_indices must be non-empty."

        # empty list is our convention for using all hap IDs
        if hap_ids_indices is None:
            hap_ids_indices = []

        self.base_type.read(
            self,
            max_rows=max_rows, 
            delimiter=delimiter,
            hap_ids_indices=hap_ids_indices,
            buffer_size=buffer_size,
            n_rows_hint=n_rows_hint,
            n_threads=n_threads,
        )
        
        if debug:
            assert len(self.haplotype_IDs) == len(set(self.haplotype_IDs)), "Repeated columns"

        log.info(f"Finished reading msp file: {self.filename}")
