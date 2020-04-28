#!/bin/bash

HDF5_VERSION='hdf5-1.8.16'

rm -r arma_libs/*
rm -r HDF5/*

mkdir arma_libs
mkdir HDF5

ARMADILLO_VERSION='armadillo-4.000.0'

rm -r $HDF5_VERSION$"/"

rm -r $ARMADILLO_VERSION$"/"
tar -xvf $ARMADILLO_VERSION$".tar.gz"

sed -i 's/unset(CMAKE_INSTALL_PREFIX)/# unset(CMAKE_INSTALL_PREFIX)/g' $ARMADILLO_VERSION$"/CMakeLists.txt"
sed -i 's/unset(INSTALL_LIB_DIR)/# unset(INSTALL_LIB_DIR)/g' $ARMADILLO_VERSION$"/CMakeLists.txt"

cd $ARMADILLO_VERSION$"/"

cmake .

make

if [ $? -eq 0 ] ; then

make install DESTDIR=../arma_libs

cd ..\

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
NEW_PATH_ARMA=$DIR$'/arma_libs/usr/lib64'
NEW_PATH_ARMA=$DIR$'/arma_libs/usr/lib'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEW_PATH_HDF5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEW_PATH_ARMA
sed -i 's/\/\/ #define ARMA_USE_CXX11/#define ARMA_USE_CXX11/g' arma_libs/usr/include/armadillo_bits/config.hpp
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
