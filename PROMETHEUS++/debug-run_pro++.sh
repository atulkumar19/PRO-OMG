#!/bin/bash

REPO_DIR=/home/nfc/myRepos/ldrdPrometheus-Upgrade
HDF5_INSTALL=/home/nfc/myRepos/ldrdPrometheus-Upgrade/HDF5/lib
ARMADILLO_INSTALL=/home/nfc/myRepos/ldrdPrometheus-Upgrade/arma_libs/lib
FORGE_PATH=/home/nfc/arm/forge/bin

# Simulation ID
ID=""

# Dimensionality of simulation
DIMENSIONALITY="1-D"

# Available number of cores in system
NUM_CORES=32

# Number of MPI processes
NUM_MPI_PROCESSES=32

# Number of OMP threads per MPI
NUM_OMP_PER_MPI=$((NUM_CORES/NUM_MPI_PROCESSES))

# Location of outputs folder
LOC_OUTPUT_FOLDER=${REPO_DIR}"/PROMETHEUS++/outputFiles"

rm -r ${LOC_OUTPUT_FOLDER}"/"${ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_INSTALL}:${ARMADILLO_INSTALL}
export PATH=${FORGE_PATH}:$PATH

echo "LD_LIBRARY_PATH: "${LD_LIBRARY_PATH}
echo "Number of MPI processes: "${NUM_MPI_PROCESSES}
echo "Number of OMP threads per MPI: "${NUM_OMP_PER_MPI}


if [$ID == ""]; then
    echo "USING DEFAULT INPUT FILES"
    ddt --connect mpirun --use-hwthread-cpus -np $((NUM_MPI_PROCESSES)) -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=$((NUM_OMP_PER_MPI)) bin/PROMETHEUS++ ${DIMENSIONALITY} ${LOC_OUTPUT_FOLDER}
else
    echo "USING MODIFIED INPUT FILES"
    ddt --connect mpirun --use-hwthread-cpus -np $((NUM_MPI_PROCESSES)) -x LD_LIBRARY_PATH -x OMP_NUM_THREADS=$((NUM_OMP_PER_MPI)) bin/PROMETHEUS++ ${DIMENSIONALITY} ${LOC_OUTPUT_FOLDER} ${ID}
fi
