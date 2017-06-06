#!/bin/bash

make clean

rm src/*~
clear

make info
make -B -j5

install_name_tool -change libarmadillo.4.dylib ../arma_libs/usr/lib/libarmadillo.4.dylib bin/PROMETHEUS++
