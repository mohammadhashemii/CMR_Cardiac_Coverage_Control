
#!/bin/sh
#PBS -l nodes=1:ppn=4:gpuen
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err
#PBS -N Saliency_detection
#PBS -q gpu_queue
#PBS -l walltime=10:00:00
#PBS -W group_list=gpu_grp -A gpu_grp
source /share/Application/Anaconda/anaconda3/SourceBio.sh
source activate saliency_detection
cd $PBS_O_WORKDIR
########     Running     #######
export HDF5_USE_FILE_LOCKING='FALSE'
python3.9 train.py
