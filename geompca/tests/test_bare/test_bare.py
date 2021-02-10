import re
import shutil

import numpy as np
from pysisyphus.helpers import geom_loader

from geompca.main import do_pca


def from_crest(log, conformers_fn="crest_conformers.xyz"):
    dihedral_re = re.compile("DIHEDRAL ANGLES as descriptors:(.+)Performing SVD", re.DOTALL)

    mobj = dihedral_re.search(log)
    dihedrals = mobj[1].strip().split("\n")
    dihedrals = np.array([line.strip().split()[1:] for line in dihedrals], dtype=int)
    dihedrals -= 1

    tmp_fn = "tmp.trj"
    shutil.copy(conformers_fn, tmp_fn)
    geoms = geom_loader(tmp_fn)
    print(len(geoms))
    energies = np.array([float(geom._comment) for geom in geoms])
    energies -= energies.min()
    energies *= 2625.499

    inds = dihedrals
    xyzs = [geom.coords3d for geom in geoms]
    labels = list(map(str, range(len(geoms))))
    do_pca(xyzs, inds, energies, labels)


def test_bare():
    with open("crest.log") as handle:
        log = handle.read()

    from_crest(log)
