#!/bin/bash
make clean

make info
make -j4

#install_name_tool -change libarmadillo.4.dylib ../arma_libs/usr/lib/libarmadillo.4.dylib bin/PROMETHEUS++
