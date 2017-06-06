#!/bin/bash

USING_C11_STANDARD='yes'
HDF5_VERSION='hdf5-1.8.16'

rm -r arma_libs/*
rm -r HDF5/*

mkdir arma_libs
mkdir HDF5

export CC=/usr/local/bin/gcc
export CPP=/usr/local/bin/cpp
export CXX=/usr/local/bin/g++
export FC=/usr/local/bin/gfortran

if [ "$USING_C11_STANDARD" == "yes" ]
then

ARMADILLO_VERSION='armadillo-4.000.0'
sed -i '.backups' 's/CCXXFLAGS=-std=c++0x/CCXXFLAGS =-std=c++11/g' PROMETHEUS++_MPI/makefile
sed -i '.backups' 's/\/\/ arma_rng::set_seed_random();/arma_rng::set_seed_random();/g' PROMETHEUS++_MPI/src/initialize.cpp

else

ARMADILLO_VERSION='armadillo-3.900.6'
sed -i '.backups' 's/CCXXFLAGS=-std=c++11/CCXXFLAGS=-std=c++0x/g' PROMETHEUS++_MPI/makefile
sed -i '.backups' 's/arma_rng::set_seed_random();/\/\/ arma_rng::set_seed_random();/g' PROMETHEUS++_MPI/src/initialize.cpp

fi

rm -r $HDF5_VERSION$"/"

rm -r $ARMADILLO_VERSION$"/"
tar -xvf $ARMADILLO_VERSION$".tar.gz"

sed -i '.backups' 's/unset(CMAKE_INSTALL_PREFIX)/# unset(CMAKE_INSTALL_PREFIX)/g' $ARMADILLO_VERSION$"/CMakeLists.txt"
sed -i '.backups' 's/unset(INSTALL_LIB_DIR)/# unset(INSTALL_LIB_DIR)/g' $ARMADILLO_VERSION$"/CMakeLists.txt"

cd $ARMADILLO_VERSION$"/"

cmake .

make

if [ $? -eq 0 ] ; then

make install DESTDIR=../arma_libs

cd ../

rm -r $ARMADILLO_VERSION$"/"

DIR=${PWD}
PREFIX=$DIR$'/HDF5'

tar -xvf $HDF5_VERSION$".tar"

cd $HDF5_VERSION

./configure --prefix=$PREFIX --enable-cxx --enable-production

make

if [ $? -eq 0 ] ; then
make check
else
echo 'HDF5 installation error: There was an error while doing "make"'
exit
fi

if [ $? -eq 0 ] ; then
make install prefix=$PREFIX
else
echo 'HDF5 installation error: There was an error while doing "make check"'
exit
fi

if [ $? -eq 0 ] ; then
make check-install
else
echo 'HDF5 installation error: There was an error while doing "make install"'
exit
fi

if [ $? -eq 0 ] ; then
cd ..
rm -r $HDF5_VERSION$"/"

NEW_PATH_HDF5=$PREFIX$'/lib'
NEW_PATH_ARMA=$DIR$'/arma_libs/usr/lib'
NEW_PATH_ARMA=$NEW_PATH_ARMA:$DIR$'/arma_libs/usr/lib'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEW_PATH_HDF5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEW_PATH_ARMA
sed -i '.backups' 's/\/\/ #define ARMA_USE_CXX11/#define ARMA_USE_CXX11/g' arma_libs/usr/include/armadillo_bits/config.hpp
echo '* * * * * * * * * * * * * * * * * * * * * *'
echo '*                                         *'
echo '*          Installation succeeded          *'
echo '*                                         *'
echo '* * * * * * * * * * * * * * * * * * * * * *'
else
echo 'HDF5 installation error: There was an error while doing "make install"'
exit
fi


else
echo 'Armadillo installation error: There was an error while doing "make install"'
exit
fi
