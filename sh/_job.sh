#!/bin/sh 
#PBS -q h-lecture
#PBS -W group_list=gt36
#PBS -l select=1:mpiprocs=1:ompthreads=4
#PBS -l walltime=00:10:00

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module load cuda pgi

# cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nvprof ./../bin/main >./../out/out.log 2>./../out/err.log