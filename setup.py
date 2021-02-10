from setuptools import find_packages, setup
import sys

if sys.version_info.major < 3:
    raise SystemExit("Python 3 is required!")

setup(
    name="geompca",
    version=0.1,
    description="PCA with a subset of internal coordinates",
    url="https://github.com/eljost/geompca",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    platforms=["unix"],
    packages=find_packages(),
    install_requires=[
        "array_to_latex",
        "joblib",
        "matplotlib",
        "numpy",
        "pysisyphus",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "geompca = geompca.geompca:run",
        ]
    },
)
