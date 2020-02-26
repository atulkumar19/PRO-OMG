#!/bin/bash

# Path to installation folder
# Change this folder to your installation folder of PROMETHEUS++
ROOT="/home/leo/Documents/PRO++"


# Paths to external libraries
HDF5_PATH=${ROOT}"/HDF5/lib"
ARMA_PATH=${ROOT}"/arma_libs/usr/lib64"

# Number of MPI processes
NUM_MPI_PROCESSES=2

# Location of outputs folder
LOC_OUTPUT_FOLDER=${ROOT}"/PROMETHEUS++/outputFiles"

# Identifier for input file. If none used, default input files used.
# FILE_ID="dispersion_relation"
# FILE_ID="GC"
# FILE_ID="warm_plasma"

rm -r ${LOC_OUTPUT_FOLDER}"/"${FILE_ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_PATH}:${ARMA_PATH}

# mpirun -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=2 -np $((NUM_MPI_PROCESSES))  bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER} ${FILE_ID}
mpirun -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=2 -np $((NUM_MPI_PROCESSES))  bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER}
