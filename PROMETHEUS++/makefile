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
	INCLUDE_HDF5 = -I$(HDF_INSTALL)/include
	INCLUDES =	-I../arma_libs/usr/include	-I../arma_libs/usr/lib
	EXTLIB = -L$(HDF_INSTALL)/lib
	LIBSHDF   = $(EXTLIB) $(HDF_INSTALL)/lib/libhdf5.a 
	LIB_FLAGS =	-L../arma_libs/usr/lib
	LIB	=	-lz	-ldl	-lm	-lhdf5_cpp	-larmadillo -lgfortran
	OMP_LIB_FLAGS =	-lgomp
endif

ifeq ($(UNAME), Linux)
	CC=mpic++

	CCXXFLAGS =-std=c++11
	CCDEFINEFLAGS=-DONED	-DHDF5_FLOAT	-DHDF5_SINGLE_FILE
	CCFLAGS =	$(CCXXFLAGS)	-fopenmp	-O2

	HDF_INSTALL = ../HDF5

	INCLUDE_HDF5	=	-I$(HDF_INSTALL)/include
	INCLUDES =	-I../arma_libs/usr/include	-I../arma_libs/usr/lib	-I../arma_libs/usr/lib64
	EXTLIB	= -L$(HDF_INSTALL)/lib
	LIBSHDF	= $(EXTLIB) $(HDF_INSTALL)/lib/libhdf5.a 	
	LIB_FLAGS =	-L../arma_libs/usr/lib -L../arma_libs/usr/lib64
	LIB	=	-lz	-ldl	-lm	-lhdf5_cpp	-larmadillo
	OMP_LIB_FLAGS =	-lgomp
endif

BINARIES = $(addprefix $(OBJ)/,main.o	\
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


info:
#	@echo	"Compiling the code for MacOS"
	@echo	$(UNAME)
	@echo	"COMPILING..."

clean:
	@echo	"CLEANING COMPILATION..."
	rm -f $(BIN)/PROMETHEUS++	$(OBJ)/*.o
