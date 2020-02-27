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


ifeq ($(UNAME), Darwin)
    CC=/Users/Leopo/Documents/merging_code/local_mpi/bin/mpic++
	CCXXFLAGS =-std=c++11
	CCDEFINEFLAGS=-DONED		-DHDF5_FLOAT		-DHDF5_SINGLE_FILE
	CCFLAGS = $(CCXXFLAGS)    -fopenmp    -O2

	HDF_INSTALL = ../HDF5
	# ARMA_LIBS_INSTALL = ../arma_libs_9
	ARMA_LIBS_INSTALL = ../arma_libs/usr

	INCLUDE_HDF5 = -I$(HDF_INSTALL)/include
	INCLUDES =	-I$(ARMA_LIBS_INSTALL)/include	-I$(ARMA_LIBS_INSTALL)/lib
	EXTLIB = -L$(HDF_INSTALL)/lib
	LIBSHDF   = $(EXTLIB) $(HDF_INSTALL)/lib/libhdf5.a
	LIB_FLAGS =	-L$(ARMA_LIBS_INSTALL)/lib
	LIB	=	-lz	-ldl	-lm	-lhdf5_cpp	-larmadillo -lgfortran
	OMP_LIB_FLAGS =	-lgomp
endif

ifeq ($(UNAME), Linux)
	CC=mpic++

	CCXXFLAGS =-std=c++11
	CCDEFINEFLAGS=-DONED	-DHDF5_FLOAT	-DHDF5_SINGLE_FILE	-DCHECKS_ON
	CCFLAGS =	$(CCXXFLAGS)	-fopenmp	-O2

	HDF_INSTALL = ../HDF5
	# ARMA_LIBS_INSTALL = ../arma_libs_9
	ARMA_LIBS_INSTALL = ../arma_libs/usr

	INCLUDE_HDF5	=	-I$(HDF_INSTALL)/include
	INCLUDES =	-I$(ARMA_LIBS_INSTALL)/include	-I$(ARMA_LIBS_INSTALL)/lib	-I$(ARMA_LIBS_INSTALL)/lib64
	EXTLIB	= -L$(HDF_INSTALL)/lib
	LIBSHDF	= $(EXTLIB) $(HDF_INSTALL)/lib/libhdf5.a
	LIB_FLAGS =	-L$(ARMA_LIBS_INSTALL)/lib -L$(ARMA_LIBS_INSTALL)/lib64
	LIB	=	-lz	-ldl	-lm	-lhdf5_cpp	-larmadillo
	OMP_LIB_FLAGS =	-lgomp
endif

BINARIES = $(addprefix $(OBJ)/,structures.o\
main.o	\
timeSteppingMethods.o	\
PIC.o	\
initialize.o	\
units.o	\
boundaryConditions.o	\
emf.o	\
generalFunctions.o	\
types.o	\
outputHDF5.o	\
alfvenic.o	\
mpi_main.o	\
randomStart.o	\
quietStart.o)

all: $(BIN)/PROMETHEUS++

$(OBJ)/%.o : $(SRC)/%.cpp
	$(CC)	-c	$(CCFLAGS)	$(CCDEFINEFLAGS)	$<	$(INCLUDES)	$(INCLUDE_HDF5)	$(LIB_FLAGS) -o $@

$(BIN)/PROMETHEUS++: $(BINARIES)
	$(CC)	$(BINARIES)	-o	$@	$(LIB_FLAGS)	$(LIBSHDF)	$(LIB)	$(OMP_LIB_FLAGS)



BINARIES = $(addprefix $(OBJ)/,structures.o\
main_dev_test.o	\
dev_test_2D.o \
types.o	\
mpi_main.o)

test: $(BIN)/PRO++TEST

$(OBJ)/%.o : $(SRC)/%.cpp
	$(CC)	-c	$(CCFLAGS)	$(CCDEFINEFLAGS)	$<	$(INCLUDES)	$(INCLUDE_HDF5)	$(LIB_FLAGS) -o $@

$(BIN)/PRO++TEST: $(BINARIES)
	$(CC)	$(BINARIES)	-o	$@	$(LIB_FLAGS)	$(LIBSHDF)	$(LIB)	$(OMP_LIB_FLAGS)


info:
	@echo	"COMPILING IN: "$(UNAME)

clean:
	@echo	"CLEANING PREVIOUS COMPILATION"
	rm -f $(BIN)/PROMETHEUS++	$(OBJ)/*.o	$(SRC)/*~
