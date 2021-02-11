import re
import shutil

import numpy as np
from pysisyphus.helpers import geom_loader
from pysisyphus.xyzloader import write_geoms_to_trj
from sklearn.cluster import KMeans

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
    xyzs = np.array([geom.coords3d for geom in geoms])
    labels = list(map(str, range(len(geoms))))
    X_new = do_pca(xyzs, inds, energies, labels)

    n_clusters = 5
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(X_new)
    cluster_inds = kmeans.labels_
    clusters = dict()
    for i, cl in enumerate(cluster_inds):
        clusters.setdefault(cl, list()).append(geoms[i])

    per_cluster = list()
    comments = list()
    for k, v in clusters.items():
        fn = f"cluster_{k:02d}.trj"
        comment = f"cluster {k}"
        comments.extend([comment for g in v])
        write_geoms_to_trj(v, fn)
        per_cluster.extend(v)
    write_geoms_to_trj(per_cluster, "clustered.trj", comments=comments)


def test_bare():
    with open("crest.log") as handle:
        log = handle.read()

    from_crest(log)
