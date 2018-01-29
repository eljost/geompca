#!/usr/bin/env python3

import argparse
from itertools import combinations
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

from qchelper.geometry import parse_trj_file, parse_xyz_file


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("trj",
                        help=".trj file with all input geometries.")
    parser.add_argument("inds",
                        help="File with the internal coordinate indices to "
                             "use for the PCA")
    parser.add_argument("--energies",
                        help="File with energies of the conformers.")
    parser.add_argument("--kcal2kj", action="store_true")
    # Anzeige von 2D, 3D, Skree und Crossplot optional machen...
    return parser.parse_args(args)


def calc_bond(xyz, ind1, ind2):
    """Calculate a bond length between two atoms."""
    return np.linalg.norm(xyz[ind1]-xyz[ind2])


def calc_angle(xyz, ind1, ind2, ind3):
    """Calculate an angle between three atoms."""
    vec1 = xyz[ind1] - xyz[ind2]
    vec2 = xyz[ind3] - xyz[ind2]
    vec1n = np.linalg.norm(vec1)
    vec2n = np.linalg.norm(vec2)
    dotp = np.dot(vec1, vec2)
    radians = np.arccos(dotp / (vec1n * vec2n))
    return np.degrees(radians)


def calc_dihedral(xyz, ind1, ind2, ind3, ind4):
    """Calculate the dihedral angle for four atoms."""
    vec1 = xyz[ind1] - xyz[ind2]
    vec2 = xyz[ind2] - xyz[ind3]
    vec3 = xyz[ind3] - xyz[ind4]

    n1 = np.cross(vec1, vec2) 
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(vec2, vec3) 
    n2 /= np.linalg.norm(n2)

    u1 = n2
    u3 = vec2/np.linalg.norm(vec2)
    u2 = np.cross(u3, u1)

    radians = -np.arctan2(n1.dot(u2), n1.dot(u1))
    return np.degrees(radians)


def calc_params(xyz, inds_list):
    calc_dict = {
        2: calc_bond,
        3: calc_angle,
        4: calc_dihedral
    }
    return [calc_dict[len(inds)](xyz, *inds) for inds in inds_list]


def skree_plot(pca):
    explained = pca.explained_variance_ratio_
    explained_cumsum = np.cumsum(explained)
    fig, ax = plt.subplots()
    xs = np.arange(1, len(explained)+1)
    ax.plot(xs, explained, "X-")
    ax.plot(xs, explained_cumsum, "X-")
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Explained variance / %")
    return ax


def pc_label(pca, i):
    return f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})"


def two_components(X_new, energies, pca):
    fig, ax = plt.subplots()
    scatter2d = ax.scatter(X_new[:,0], X_new[:,1], c=energies)
    fig.colorbar(scatter2d, label="kJ/mol")
    for i in range(energies.size):
        ax.annotate(str(i+1), xy=X_new[i,:2]+0.1)
        
    ax.set_xlabel(pc_label(pca, 0))
    ax.set_ylabel(pc_label(pca, 1))
    return ax


def three_components(X_new, energies, pca):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter3d = ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2], c=energies)
    scatter3d.set_edgecolors = scatter3d.set_facecolors = lambda *args: None
    fig.colorbar(scatter3d, label="kJ/mol")
    for i in range(energies.size):
        ax.text(*X_new[i,:3]+0.1, str(i+1))
        
    ax.set_xlabel(pc_label(pca, 0))
    ax.set_ylabel(pc_label(pca, 1))
    ax.set_zlabel(pc_label(pca, 2))
    return ax


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
            scatter2d = ax.scatter(X_new[:,i], X_new[:,j], c=energies)
                
            #ax.set_xlabel(pc_label(pca, i))
            #ax.set_ylabel(pc_label(pca, j))
    return ax_arr


def all_bond_inds(atom_num):
    atom_range = range(atom_num)
    return list(combinations(atom_range, 2))


def run():
    args = parse_args(sys.argv[1:])
    trj_fn = args.trj #"xyzs/aligned.trj"
    xyzs = [geom[1] for geom in parse_trj_file(trj_fn)]

    with open(args.inds) as handle:
        inds = handle.read()
    inds = [[int(i) for i in line.split()]
            for line in inds.strip().split("\n")]
    X = np.array([calc_params(xyz, inds) for xyz in xyzs])
    X_scaled = preprocessing.scale(X)
    #X_scaled = X # no scaling
    pca = PCA()
    X_new = pca.fit_transform(X_scaled)
    if args.energies:
        energies = np.loadtxt(args.energies)
        if args.kcal2kj:
            energies *= 4.184
    else:
        print("Couldn't find any energies!")
        energies = np.zeros(len(xyzs))

    skree_ax = skree_plot(pca)
    ax_2d = two_components(X_new, energies, pca)
    ax_3d = three_components(X_new, energies, pca)
    cross_arr = cross_components(X_new, energies, pca, 4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
