#!/bin/bash
echo "Install CUDPP"
git clone https://github.com/cudpp/cudpp.git
cd cudpp
git submodule init
git submodule update
mkdir -p build
cd build
cmake ..
make

cd ..
cd ..

export CUDPP_DIR=../cudpp

export CUDPP_LIB=$CUDPP_DIR/build/lib
export CUDPP_INCLUDE=$CUDPP_DIR/include

echo "Install MultiGraph"
cd code
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DCC -DSPARSE_MODE -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o CC
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DPR_T -DDENSE_MODE -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o DATA_DRIVEN_PR
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DPR_D -DDENSE_MODE -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o TOPOLOGY_DRIVEN_PR
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DBFS  -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o BFS
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DSSSP  -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o SSSP
nvcc -O3 -gencode arch=compute_35,code=sm_35 -DBC  -DPRINT_OUTPUT0 -DVALIDATE -DSYM -DTRACE0 -lcudpp --use_fast_math -Xptxas "-v -dlcm=cg" -I${CUDPP_INCLUDE} -L${CUDPP_LIB} read_input.cu function.cu gen_structure.cu release_structure.cu process.cu process2.cu processi.cu validate.cu main.cu -o BC
cd ..

