# 	COPYRIGHT 2015-2019 LEOPOLDO CARBAJAL

#	This file is part of PROMETHEUS++.
#
#   PROMETHEUS++ is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	any later version.

#	PROMETHEUS++ is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/>.

#Makefile
UNAME:=$(shell uname)


SRC=src
OBJ=obj
LIBS=libs
BIN=bin

MPICXX=
CCXXFLAGS=-std=c++11
CCDEFINEFLAGS=-DHDF5_FLOAT	-DHDF5_SINGLE_FILE	-DCHECKS_ON
CCFLAGS=$(CCXXFLAGS)	-fopenmp	-O3
HDF5_INSTALL=
ARMADILLO_INSTALL=

INCLUDE_HDF5=-I$(HDF5_INSTALL)/include
INCLUDES=-I$(ARMADILLO_INSTALL)/include	-I$(ARMADILLO_INSTALL)/lib	-I$(ARMADILLO_INSTALL)/lib64
EXTLIB=-L$(HDF5_INSTALL)/lib
LIBSHDF=$(EXTLIB) $(HDF5_INSTALL)/lib/libhdf5.a
LIB_FLAGS=-L$(ARMADILLO_INSTALL)/lib	-L$(ARMADILLO_INSTALL)/lib64
LIB=-lz	-ldl	-lm	-lhdf5_cpp	-larmadillo #-lgfortran
OMP_LIB_FLAGS=-lgomp


BINARIES = $(addprefix $(OBJ)/,structures.o\
main.o	\
PIC.o	\
initialize.o	\
units.o	\
boundaryConditions.o	\
fields.o	\
types.o	\
outputHDF5.o	\
mpi_main.o	\
randomStart.o	\
quietStart.o)

all: $(BIN)/PROMETHEUS++

$(OBJ)/%.o : $(SRC)/%.cpp
	$(MPICXX)	-c	$(CCFLAGS)	$(CCDEFINEFLAGS)	$<	$(INCLUDES)	$(INCLUDE_HDF5)	$(LIB_FLAGS) -o $@

$(BIN)/PROMETHEUS++: $(BINARIES)
	$(MPICXX)	$(BINARIES)	-o	$@	$(LIB_FLAGS)	$(LIBSHDF)	$(LIB)	$(OMP_LIB_FLAGS)



BINARIES_DEV = $(addprefix $(OBJ)/,structures.o\
main_dev_test.o	\
dev_test_2D.o \
types.o	\
mpi_main.o)

test: $(BIN)/PRO++TEST

$(OBJ)/%.o : $(SRC)/%.cpp
	$(MPICXX)	-c	$(CCFLAGS)	$(CCDEFINEFLAGS)	$<	$(INCLUDES)	$(INCLUDE_HDF5)	$(LIB_FLAGS) -o $@

$(BIN)/PRO++TEST: $(BINARIES_DEV)
	$(MPICXX)	$(BINARIES_DEV)	-o	$@	$(LIB_FLAGS)	$(LIBSHDF)	$(LIB)	$(OMP_LIB_FLAGS)


info:
	@echo	"COMPILING IN: "$(UNAME)

clean:
	@echo	"CLEANING PREVIOUS COMPILATION"
	rm -f $(BIN)/PROMETHEUS++	$(OBJ)/*.o	$(SRC)/*~
