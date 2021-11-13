#!/bin/sh
#PBS -l nodes=1:ppn=4
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err
#PBS -N Saliency_detection
#PBS -q batch
#PBS -l walltime=10:00:00
cd $PBS_O_WORKDIR
python3.9 KFold.py
