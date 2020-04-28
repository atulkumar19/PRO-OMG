#!/bin/bash

REPO_DIR=
HDF5_INSTALL=
ARMADILLO_INSTALL=

# Available number of cores in system
NUM_CORES=4

# Number of MPI processes
NUM_MPI_PROCESSES=2

# Number of OMP threads per MPI
NUM_OMP_PER_MPI=$((NUM_CORES/NUM_MPI_PROCESSES))

# Location of outputs folder
LOC_OUTPUT_FOLDER=${REPO_DIR}"/PROMETHEUS++/outputFiles"

rm -r ${LOC_OUTPUT_FOLDER}"/"${FILE_ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_INSTALL}:${ARMADILLO_INSTALL}

echo "Number of MPI processes: "${NUM_MPI_PROCESSES}
echo "Number of OMP threads per MPI: "${NUM_OMP_PER_MPI}

mpirun -np $((NUM_MPI_PROCESSES)) -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=$((NUM_OMP_PER_MPI)) bin/PROMETHEUS++ ${LOC_OUTPUT_FOLDER}
