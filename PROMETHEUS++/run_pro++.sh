#!/bin/bash

# Path to installation folder
# Change this folder to your installation folder of PROMETHEUS++
ROOT="/home/leo/Documents/PRO++"


# Paths to external libraries
HDF5_PATH=${ROOT}"/HDF5/lib"
ARMA_PATH=${ROOT}"/arma_libs/usr/lib64"

# Available number of cores in system
NUM_CORES=4

# Number of MPI processes
NUM_MPI_PROCESSES=2

# Number of OMP threads per MPI
NUM_OMP_PER_MPI=$((NUM_CORES/NUM_MPI_PROCESSES))

# Location of outputs folder
LOC_OUTPUT_FOLDER=${ROOT}"/PROMETHEUS++/outputFiles"

rm -r ${LOC_OUTPUT_FOLDER}"/"${FILE_ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_PATH}:${ARMA_PATH}

echo "Number of MPI processes: "${NUM_MPI_PROCESSES}
echo "Number of OMP threads per MPI: "${NUM_OMP_PER_MPI}

mpirun -np $((NUM_MPI_PROCESSES)) -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=$((NUM_OMP_PER_MPI)) bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER}
