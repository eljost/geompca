import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
from pysisyphus.helpers import geom_loader
from pysisyphus.xyzloader import write_geoms_to_trj
from sklearn.cluster import KMeans

from geompca.main import do_pca


def sort_into_clusters(geoms, cluster_inds):
    clusters = dict()
    for i, cl in enumerate(cluster_inds):
        clusters.setdefault(cl, list()).append(geoms[i])
    return clusters


def write_clusters(clusters, prefix=""):
    if prefix != "":
        prefix = prefix + "_"
    per_cluster = list()
    comments = list()
    for k, v in clusters.items():
        fn = f"{prefix}cluster_{k:02d}.trj"
        comment = f"cluster {k}"
        comments.extend([comment for g in v])
        write_geoms_to_trj(v, fn)
        per_cluster.extend(v)
    write_geoms_to_trj(per_cluster, f"{prefix}clustered.trj", comments=comments)


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

    crest_cluster_inds = np.loadtxt("cluster.order", skiprows=1)[:,-1].astype(int)
    crest_clusters = sort_into_clusters(geoms, crest_cluster_inds)
    write_clusters(crest_clusters, "crest")

    energies = np.array([float(geom._comment) for geom in geoms])
    energies -= energies.min()
    energies *= 2625.499

    inds = dihedrals
    xyzs = np.array([geom.coords3d for geom in geoms])
    labels = list(map(str, range(len(geoms))))
    X_new = do_pca(xyzs, inds, energies, labels)

    n_clusters = 7
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(X_new)
    cluster_inds = kmeans.labels_
    clusters = sort_into_clusters(geoms, cluster_inds)
    write_clusters(clusters)

    print(f"Sorted PCs in {n_clusters} clusters")
    for k, v in clusters.items():
        print(f"\tCluster {k:02d}: {len(v): >8d} entries")
    print()


    inertias = list()
    for k in range(1, 10):
        kmeans = KMeans(init="k-means++", n_clusters=k, n_init=4)
        kmeans.fit(X_new[:,:4])
        inertias.append(kmeans.inertia_)
    plt.plot(inertias, "o-")
    plt.show()


def test_bare():
    with open("crest.log") as handle:
        log = handle.read()

    from_crest(log)
