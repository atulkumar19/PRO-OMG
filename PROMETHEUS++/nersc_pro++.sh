#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 1  ###number of nodes
###SBATCH -S 36  
#SBATCH -C haswell
#SBATCH -t 00:30:00
###SBATCH -L SCRATCH ###need scratch access for this job
#SBATCH -A m3728   ##### Add your project ID here
#SBATCH --mail-user=kumara1@ornl.gov
#SBATCH --mail-type=ALL

## fix formatted output in cray env
if [ "$PE_ENV" == "CRAY" ]; then
  export FILENV=my_filenenv
  assign -U on g:sf
fi
#OpenMP settings
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

###TO run PRO++OMG########################################
REPO_DIR=~/PRO-OMG
HDF5_INSTALL=~/PRO-OMG/HDF5/lib
ARMADILLO_INSTALL=~/PRO-OMG/arma_libs/lib64

# Simulation ID
ID=""

# Dimensionality of simulation
DIMENSIONALITY="1-D"

# Location of outputs folder
LOC_OUTPUT_FOLDER=${REPO_DIR}"/PROMETHEUS++/outputFiles"

rm -r ${LOC_OUTPUT_FOLDER}"/"${ID}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HDF5_INSTALL}:${ARMADILLO_INSTALL}

echo "LD_LIBRARY_PATH: "${LD_LIBRARY}



echo "USING DEFAULT INPUT FILES"
srun   -n 4 bin/PROMETHEUS++ ${DIMENSIONALITY} ${LOC_OUTPUT_FOLDER}

cd   inputFiles/
cp input_file.input ions_properties.ion  *.txt ../outputFiles/
cd ..
cd outputFiles/
git log --oneline -1 > commitHash.txt


