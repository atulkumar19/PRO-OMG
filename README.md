# PROMETHEUS++: Hybrid code for fusion and space simulations #

# IMPORTANT! #

Do not cick on revoke in the Access level option!. This will remove your permissions for working on PROMETHEUS++.

# Getting started #

1. Install git in your computer.

2. Move to the directory in your computer where the installation of PROMETHEUS++ will be done.

3. Clone the respository to your computer. Use the option Clone on the top left-hand corner of this page.

4. A window will pop up showing two options: HTTPS and SSH. You will get something similar to: "git clone https://your_bitbucket_user@bitbucket.org/lcarbajal/prometheus_dev.git"

5. Copy this link in your Terminal. If you want to rename the directory you can do this by including the name of the new directory as follows: "git clone https://your_bitbucket_user@bitbucket.org/lcarbajal/prometheus_dev.git new_name"

6. You will need to introduce your password.

7. cd to your new folder.

# How to install #

Make sure you have at least half an hour for the installation, it will take some time! This due to the installation of the HDF5 library.

### What do you need in your computer before installing PROMETHEUS++? ###

It is important to have the following libraries and programs in your system in order to install PROMETHEUS++. Using the software manager of your Linux (Mac) system to get them is always the best option.

* [CMake](http://www.cmake.org/)
* You will need mainly two libraries for installing Armadillo C++. [BLAS](https://launchpad.net/ubuntu/precise/+package/libblas-dev) and [LAPACK](http://packages.ubuntu.com/source/lucid/lapack), you are able to use both, static or shared versions of these libraries (it is up to you).
* [ATLAS](http://math-atlas.sourceforge.net/)
* [Boost](http://www.boost.org/)
* [zlib](http://zlib.net/)

PROMETHEUS++ has been successfully compiled on **Mac OS 10.6.X and 10.7.X** and on Linux distributions as **Ubuntu** and **Suse Linux**. However, there are some know problems when using versions of the [GNU Compiller Collection](http://gcc.gnu.org/) higher than 4.4 when installing the HDF5 libraries. **Users are highly encouraged to follow these instructions to avoid problems during the installation:**

1. Open install_external_libraries.sh with a text editor and change the option USING_C11_STANDARD to 'yes' or 'no'. This depending if your compiler support the [C++11 standard](http://www.cprogramming.com/c++11/what-is-c++0x.html) or not.

2. Check your versions of C and C++ compilers. To do so, type '*gcc --version*' and '*g++ --version*' on your Terminal.

3. If your versions are higher than 4.4, you have to install the 4.4 versions, too. **Do not uninstall your current versions**, you can install the 4.4 versions along with any other versions on your computer.

4. If you had to install the 4.4 versions of the GNU compilers, then change the line "*CC='gcc -m64' ./configure --prefix=$PREFIX --enable-cxx --enable-production*" to "*CC='gcc-4.4 -m64' ./configure --prefix=$PREFIX --enable-cxx --enable-production*" in the install_external_libraries.sh file.

5. Type the following command on the Terminal you are currently working on "*. ./install_external_libraries.sh*". **Note the space between the two dots**, this is necessary if you want the installation to be done.

6. Your installation will be complete when you get the message "Installation succeeded".

# Problems when installing PROMETHEUS++? #

Please report any problem to **L.Carbajal-Gomez (at) warwick.ac.uk**. I will try to reply ASAP to your queries.

# Running PROMETHEUS++ #

mpirun -np (even number of MPI processes) bin/PROMETHEUS++ (folder that will contain outputs folder) (name of outputs folder)