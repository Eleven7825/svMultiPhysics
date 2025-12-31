ls /petsc/lib/petsc/conf
export PETSC_DIR=/petsc
export PETSC_ARCH=
echo $PETSC_DIR
echo $PETSC_ARCH
cd /svfsi
rm -rf svFSI-prefix svFSI-build CMakeCache.txt CMakeFiles
cmake -DSV_USE_PETSC=ON .
make -j4
