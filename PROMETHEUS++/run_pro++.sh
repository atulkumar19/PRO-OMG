#!/bin/bash

# Paths to external libraries
HDF5_PATH="/home/leo/Documents/PRO++/HDF5/lib"
#ARMA_PATH="/home/leo/Documents/PRO++/arma_libs/usr/lib64"
ARMA_PATH="/home/leo/Documents/PRO++/arma_libs_9/lib"

# Number of MPI processes
NUM_MPI_PROCESSES=2

# Location of outputs folder
# LOC_OUTPUT_FOLDER="/home/leo/Documents/PRO++/PROMETHEUS++/Tests"
LOC_OUTPUT_FOLDER="/home/leo/Documents/PRO++/PROMETHEUS++/outputFiles"

# File identifier
# FILE_ID="dispersion_relation"
# FILE_ID="GC"
# FILE_ID="warm_plasma"

rm -r ${LOC_OUTPUT_FOLDER}"/"${FILE_ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_PATH}:${ARMA_PATH}

# mpirun -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=2 -np $((NUM_MPI_PROCESSES))  bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER} ${FILE_ID}
mpirun -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=2 -np $((NUM_MPI_PROCESSES))  bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER}
