#!/bin/bash
make clean

make info

make all

SYS=$(uname)

if [ ${SYS} = "Darwin" ]; then
    echo "Additional steps for compilation on OS... DONE!"
    install_name_tool -change libarmadillo.4.dylib ARMA_LIBS/usr/lib/libarmadillo.4.dylib bin/PROMETHEUS++
else
    echo "Not aditional steps are needed at this time... "
fi
