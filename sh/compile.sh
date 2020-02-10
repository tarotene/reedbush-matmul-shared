#!/bin/bash

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nvcc -gencode arch=compute_60,code=sm_60 ./../src/main.cu -o ./../bin/main
# nvcc -c -Xcompiler -fopenmp -gencode arch=compute_60,code=sm_60 ./../src/main.cu -o ./../bin/main