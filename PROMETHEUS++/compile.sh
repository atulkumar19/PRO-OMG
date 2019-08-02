!/bin/bash
#make clean

make info
make -j4

#UNAME=$(uname)
#UNAME="Linux"
#if [$(UNAME) == "Darwing"]; then
#    install_name_tool -change libarmadillo.4.dylib ../arma_libs/usr/lib/libarmadillo.4.dylib bin/PROMETHEUS++
#    echo "Additional steps for compilation on OS... DONE!"
#fi
