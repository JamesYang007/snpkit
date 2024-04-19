import snpkit as sk
import snpkit.io
import numpy as np
import os


def test_msp_reader():
    n_samples = 1000
    n_rows = 20
    filename = "/tmp/test_1.msp"

    with open(filename, "w") as f:
        f.write(
            "#Subpopulation order/codes: A=0\tB=1\tC=2\tD=3\n"
        )
        sample_ids = np.arange(0, n_samples)
        hap_ids = []
        for i in sample_ids:
            hap_ids.append(str(i) + ".0")
            hap_ids.append(str(i) + ".1")
        f.write(
            "#chm\tspos\tepos\tsgpos\tegpos\tn snps\t" +
            "\t".join(hap_ids) + 
            "\n"
        )
        chm = np.full(n_rows, 1)
        spos = np.random.randint(0, 100, n_rows)
        epos = np.random.randint(0, 100, n_rows)
        sgpos = np.random.normal(0, 1, n_rows)
        egpos = np.random.normal(0, 1, n_rows)
        n_snps = np.random.randint(0, 100, n_rows)
        lai = np.random.randint(0, 4, (n_rows, len(hap_ids)))
        for i in range(n_rows):
            lai_str = '\t'.join([str(x) for x in lai[i]])
            f.write(
                f"{chm[i]}\t{spos[i]}\t{epos[i]}\t{sgpos[i]}\t{egpos[i]}\t{n_snps[i]}\t{lai_str}\n"
            )

    handle = sk.io.MSPReader(filename)
    handle.read(debug=True)

    # check ancestry map
    amap = handle.ancestry_map
    assert len(amap) == 4
    assert amap["A"] == 0
    assert amap["B"] == 1
    assert amap["C"] == 2
    assert amap["D"] == 3

    # check hap IIDs
    assert np.all(handle.haplotype_IDs == hap_ids)

    # check sample IIDs
    assert np.all([int(iid) for iid in handle.sample_IDs] == sample_ids)

    assert np.allclose(handle.chm, chm)
    assert np.allclose(handle.pos[:, 0], spos)
    assert np.allclose(handle.pos[:, 1], epos)
    assert np.allclose(handle.gpos[:, 0], sgpos)
    assert np.allclose(handle.gpos[:, 1], egpos)
    assert np.allclose(handle.n_snps, n_snps)
    assert np.allclose(handle.lai, lai)

    os.remove(filename)
