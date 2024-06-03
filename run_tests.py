import numpy as np
import pandas as pd
import sys
import os

# ==============================================================================
pathname = '/mnt/home/cottre61'
tests_path = '/mnt/home/cottre61/TAST/tests'

data_list = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']

for data in data_list:
    if os.path.exists(pathname):
        job_name = 'dlpfc_%s'%(data)
        filename = '/mnt/home/cottre61/TAST/tests/dlpfc_%s.sb'%(data)
        qsubfile = open(filename, "w")
        qsubfile.write("#!/bin/bash \n")
        qsubfile.write("#SBATCH --time=12:00:00 \n")
        qsubfile.write("#SBATCH --ntasks-per-node=1 \n")
        qsubfile.write("#SBATCH --nodes=1 \n")
        qsubfile.write("#SBATCH --cpus-per-task=4 \n")
        qsubfile.write("#SBATCH --mem=60G \n")
        qsubfile.write("#SBATCH --job-name=%s \n" % job_name)
        qsubfile.write("#SBATCH --output=/mnt/home/cottre61/TAST/tests/%s.out \n"%job_name)
        qsubfile.write("module load GCC/6.4.0-2.28 OpenMPI/2.1.2-CUDA \n")
        qsubfile.write("conda activate STAGATE \n")
        qsubfile.write("cd /mnt/home/cottre61/TAST \n")
        qsubfile.write("python Visium_DLPFC.py %s \n" %(data))
        qsubfile.write("scontrol show job $SLURM_JOB_ID")
        qsubfile.close()
        os.system(f"sbatch {filename}")
