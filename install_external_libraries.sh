#!/bin/bash

REPO_DIR=${PWD}

# * * * * * Please change these variables as required * * * * *
export CC=gcc
export CPP=cpp
export CXX=g++
export FC=gfortran

MPICXX=mpic++

HDF5_INSTALLATION_FOLDER=$REPO_DIR
ARMADILLO_INSTALLATION_FOLDER=$REPO_DIR
# * * * * * Please change these variables as required * * * * *

HDF5_INSTALLATION_FOLDER=$HDF5_INSTALLATION_FOLDER"/HDF5"
ARMADILLO_INSTALLATION_FOLDER=$ARMADILLO_INSTALLATION_FOLDER"/arma_libs"

HDF5_VERSION='hdf5-1.10.4'
ARMADILLO_VERSION='armadillo-9.850.1'

# Delete any existing previous instalation of the libraries
rm -r $HDF5_INSTALLATION_FOLDER
rm -r $ARMADILLO_INSTALLATION_FOLDER

# Create new folders for local installation of HDF5 and armadillo
mkdir $HDF5_INSTALLATION_FOLDER
mkdir $ARMADILLO_INSTALLATION_FOLDER

# Delete existing HDF5 and armadillo build folders
rm -r $HDF5_VERSION
rm -r $ARMADILLO_VERSION

# Local installation of Armadillo library

# We extract the source code of armadillo library
tar -xvf $ARMADILLO_VERSION$".tar.xz"

# We enter the source code directory of armadillo
cd $ARMADILLO_VERSION

# Armadillo installation
cmake . -DCMAKE_INSTALL_PREFIX:PATH=ARMADILLO_INSTALLATION_FOLDER

make

if [ $? -eq 0 ] ; then
make install

cd ../

rm -r $ARMADILLO_VERSION
else
echo 'ERROR: Uh-oh! Something went wrong in ARMADILLO installation!'
return
fi

# Local installation of HDF5 library

# We extract the HDF5 source code
tar -xvf $HDF5_VERSION".tar.gz"

# Enter the HDF5 source code folder
cd $HDF5_VERSION

# Set the prefix for installation folder
PREFIX=$HDF5_INSTALLATION_FOLDER

./configure --prefix=$PREFIX --enable-cxx --enable-build-mode=production

make

if [ $? -eq 0 ] ; then
make install

cd ../

rm -r $HDF5_VERSION
else
echo 'ERROR: Uh-oh! Something went wrong in HDF5 installation!'
return
fi

# Setting up Makefile and Compile environment variables
sed -i 's/MPICXX=/'"MPICXX=${MPICXX}/g" PROMETHEUS++/Makefile
sed -i 's/HDF5_INSTALL=/'"HDF5_INSTALL=${HDF5_INSTALLATION_FOLDER}/g" PROMETHEUS++/Makefile
sed -i 's/ARMADILLO_INSTALL=/'"ARMADILLO_INSTALL=${ARMADILLO_INSTALLATION_FOLDER}/g" PROMETHEUS++/Makefile

sed -i 's/ARMA_LIBS/'"${ARMADILLO_INSTALLATION_FOLDER}/g" PROMETHEUS++/Makefile

echo '* * * * * * * * * * * * * * * * * * * * * *'
echo '*          Installation succeeded         *'
echo '* * * * * * * * * * * * * * * * * * * * * *'
