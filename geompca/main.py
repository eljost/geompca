import argparse
from itertools import combinations
from math import cos, sin
from pathlib import Path
import re
import sys

import array_to_latex as a2l
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import joblib

from pysisyphus.intcoords import Stretch, Bend, Torsion
from pysisyphus.xyzloader import parse_trj_file

try:
    import mymplrc
except ModuleNotFoundError:
    print("mymplrc is not installed. skipping import.")


np.set_printoptions(suppress=True, precision=4)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("trj", help=".trj file with all input geometries.")
    parser.add_argument(
        "inds", help="File with the internal coordinate indices to " "use for the PCA"
    )
    parser.add_argument("--energies", help="File with energies of the conformers.")
    parser.add_argument("--kcal2kj", action="store_true")
    parser.add_argument(
        "--auinp",
        action="store_true",
        help="Expect absolute energies in a.u. as input.",
    )
    select_group = parser.add_mutually_exclusive_group(required=False)
    select_group.add_argument(
        "--first", default=None, type=int, help="Only consider the first N geometries."
    )
    select_group.add_argument(
        "--skip", type=int, nargs="+", default=[], help="Skip these indices (0-based)."
    )
    # Anzeige von 2D, 3D, Skree und Crossplot optional machen...
    return parser.parse_args(args)


PRIM_DICT = {
    2: Stretch,
    3: Bend,
    4: Torsion,
}


def calc_params(xyz, inds_list):
    params = list()
    for inds in inds_list:
        cls = PRIM_DICT[len(inds)]
        prim_val = cls._calculate(xyz, inds)
        # See https://pubmed.ncbi.nlm.nih.gov/15521057/ for a discussion,
        # why we can't directly use dihedrals.
        # And:
        #   https://aip.scitation.org/doi/abs/10.1063/1.2746330
        if len(inds) == 4:
            cos_rad = cos(prim_val)
            sin_rad = sin(prim_val)
            params.extend((cos_rad, sin_rad))
            continue
        params.append(prim_val)
    return params


def skree_plot(pca):
    explained = pca.explained_variance_ratio_
    explained_cumsum = np.cumsum(explained)
    fig, ax = plt.subplots()
    xs = np.arange(1, len(explained) + 1)
    ax.plot(xs, explained, "X-")
    ax.plot(xs, explained_cumsum, "X-")
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Explained variance / %")
    return fig, ax


def pc_label(pca, i):
    return f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})"


def two_components(X_new, energies, pca, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter2d = ax.scatter(X_new[:, 0], X_new[:, 1], c=energies)
    fig.colorbar(scatter2d, label="E / kJ mol$^{-1}$")
    for i in range(energies.size):
        ax.annotate(labels[i], xy=X_new[i, :2] + 0.1)

    ax.set_xlabel(pc_label(pca, 0))
    ax.set_ylabel(pc_label(pca, 1))
    plt.tight_layout()
    return fig, ax


def three_components(X_new, energies, pca):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter3d = ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=energies)
    scatter3d.set_edgecolors = scatter3d.set_facecolors = lambda *args: None
    fig.colorbar(scatter3d, label="kJ/mol")
    for i in range(energies.size):
        ax.text(*X_new[i, :3] + 0.1, str(i + 1))

    ax.set_xlabel(pc_label(pca, 0))
    ax.set_ylabel(pc_label(pca, 1))
    ax.set_zlabel(pc_label(pca, 2))
    return fig, ax


def unset_ticks():
    plt.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        right="off",
        left="off",
        labelbottom="off",
        labeltop="off",
        labelright="off",
        labelleft="off",
    )


def cross_components(X_new, energies, pca, comp_num):
    fig = plt.figure()
    shape = (comp_num, comp_num)
    ax_arr = list()
    for i in range(comp_num):
        for j in range(comp_num):
            loc = (i, j)
            ax = plt.subplot2grid(shape, loc)
            ax_arr.append(ax)
            unset_ticks()
            if i == j:
                ax.text(0.25, 0.45, pc_label(pca, i), size=10)
                continue
            ax.scatter(X_new[:, i], X_new[:, j], c=energies)

            # ax.set_xlabel(pc_label(pca, i))
            # ax.set_ylabel(pc_label(pca, j))
    return fig, ax_arr


def all_bond_inds(atom_num):
    atom_range = range(atom_num)
    return list(combinations(atom_range, 2))


def rank_energies(energies):
    # import pdb; pdb.set_trace()
    print("Geoms are given with 1-based indexing.")
    for i, ind in enumerate(np.argsort(energies), 1):
        print(f"Rank {i:02d}: geom {ind+1:02d}, {energies[ind]:>5.1f} kJ/mol")
    pass


def load(X_new_fn, energies_fn, pca_fn):
    X_new = np.loadtxt(X_new_fn)
    energies = np.loadtxt(energies_fn)
    pca = joblib.load(pca_fn)
    return X_new, energies, pca


def gas_dual():
    base = Path(".").resolve()
    bare = "bare_gas_X_new.dat bare_gas_energies_kjmol.dat bare_gas_pca.pkl"
    mecoome = "mecoome_gas_X_new.dat mecoome_gas_energies_kjmol.dat mecoome_gas_pca.pkl"
    run_dual(base, bare, mecoome)


def run_dual(base, fns1, fns2):
    base = Path(base)
    data1 = [base / fn for fn in fns1.split()]
    data1 = load(*data1)
    data2 = [base / fn for fn in fns2.split()]
    data2 = load(*data2)
    # fig1, _ = two_components(*data1)
    # fig2, _ = two_components(*data2)
    fig = dual_2d(*data1, *data2)
    plt.tight_layout()
    plt.savefig("dual.svg", transparent=True)
    plt.show()


def dual_2d(X_new1, energies1, pca1, labels1, X_new2, energies2, pca2, labels2):
    """
    https://stackoverflow.com/questions/46106912
    """
    en_min = min(energies1.min(), energies2.min())
    en_max = max(energies1.max(), energies2.max())
    norm = plt.Normalize(en_min, en_max)

    def plot(ax, X_new, energies, pca, labels):
        scatter2d = ax.scatter(X_new[:, 0], X_new[:, 1], c=energies, norm=norm)
        for i in range(energies.size):
            ax.annotate(labels[i], xy=X_new[i, :2] + 0.1)
        ax.set_xlabel(pc_label(pca, 0))
        ax.set_ylabel(pc_label(pca, 1))
        return scatter2d

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5.5, 3.5))
    l1 = plot(ax1, X_new1, energies1, pca1)
    l2 = plot(ax2, X_new2, energies2, pca2)
    fig.colorbar(l1, ax=(ax1, ax2), label="kJ/mol")
    return fig


def skip_items(skip_inds, iterable):
    return [item for i, item in enumerate(iterable) if not (i in skip_inds)]


def do_pca(xyzs, inds, energies, labels):
    # Report employed internals
    print("Primitive internals:")
    for inds_ in inds:
        print(f"\t{PRIM_DICT[len(inds_)]}: {inds_}")
    print()

    # Calculate coordinate values
    X = np.array([calc_params(xyz, inds) for xyz in xyzs])
    # Variance 1, mean at 0
    X_scaled = preprocessing.scale(X)

    pca = PCA()
    X_new = pca.fit_transform(X_scaled)

    # Report eigenvalues
    print("Explained variance ratio:")
    ev_thresh = 0.05
    cumsum = 0.
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cumsum += ev
        if ev <= ev_thresh:
            print(f"\tRemaining eigenvalues are below {ev_thresh:.4f}")
            break
        print(f"\tPC {i:02d}: {ev:.4f}, Σ {cumsum:.4f}")
    print()

    joblib.dump(pca, "pca.pkl")
    np.savetxt("energies_kjmol.dat", energies)
    np.savetxt("X_new.dat", X_new)

    # Only report first two PCs
    tab = a2l.to_ltx(
        pca.components_[:2].T, frmt="{:.2f}", arraytype="tabular", print_out=False
    )

    with open("pca_comps_tabular.tex", "w") as handle:
        handle.write(tab)

    fig_skree, skree_ax = skree_plot(pca)
    fig_2d, ax_2d = two_components(X_new, energies, pca, labels)
    # import matplotlib
    # from mpl_toolkits.mplot3d import Axes3D
    # fig_3d, ax_3d = three_components(X_new, energies, pca)
    # fig_cross, cross_arr = cross_components(X_new, energies, pca, 4)

    plt.tight_layout()
    plt.show()

    fig_skree.savefig("skree.pdf")
    fig_skree.savefig("skree.svg")
    fig_2d.savefig("geompca_2d.pdf")
    fig_2d.savefig("geompca_2d.svg")
    # fig_3d.savefig("geompca_3d.pdf")
    # fig_3d.savefig("geompca_3d.svg")
    # fig_cross.savefig("geompca_cross.pdf")
    # fig_cross.savefig("geompca_cross.svg")

    return X_new


def run():
    args = parse_args(sys.argv[1:])
    trj_fn = args.trj  # "xyzs/aligned.trj"
    geoms = [geom for geom in parse_trj_file(trj_fn)]
    xyzs = [coords for atoms, coords in geoms]

    labels = [str(i) for i, _ in enumerate(xyzs, 1)]
    if args.energies:
        energies = np.loadtxt(args.energies)
        if args.kcal2kj:
            energies *= 4.184
        if args.auinp:
            energies -= energies.min()
            energies *= 2625.50
        rank_energies(energies)
    else:
        print("Couldn't find any energies!")
        energies = np.zeros(len(xyzs))

    first = args.first
    skip_inds = args.skip
    if first:
        xyzs = xyzs[: first]
        energies = energies[: first]
        labels = labels[: first]

    xyzs = skip_items(skip_inds, xyzs)
    energies = skip_items(skip_inds, energies)
    labels = skip_items(skip_inds, labels)
    if skip_inds:
        print("!" * 10)
        print(f"Skipped entrie(s): {' '.join([str(ind) for ind in args.skip])}")
        print("!" * 10)
    energies = np.array(energies)

    with open(args.inds) as handle:
        inds = handle.read()
    inds = [[int(i) for i in line.split()] for line in inds.strip().split("\n")]
    do_pca(xyzs, inds, energies, labels)
