#!/bin/bash
#SBATCH -J run1_NICER
#SBATCH -N 15
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=t.riley.phd@gmail.com
#SBATCH --mail-type=END

echo start of job in directory $SLURM_SUBMIT_DIR
echo number of nodes is $SLURM_JOB_NUM_NODES
echo the allocated nodes are:
echo $SLURM_JOB_NODELIST

module purge
module load python/2.7.14
module load intel/18.2.199
module load intelmpi/18.2
module load gsl/2.5-icc

dirname=$SLURM_JOBID
mkdir /tmpdir/$LOGNAME/$dirname
cp -r $HOME/J0740_STU /tmpdir/$LOGNAME/$dirname/.
cd /tmpdir/$LOGNAME/$dirname/J0740_STU/modules

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/MultiNest/MultiNest_v3.12_CMake/multinest/lib
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages/:$PYTHONPATH
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo directory is $PWD
srun python $HOME/J0740_STU/modules/NICER/run1/main.py @$HOME/J0740_STU/modules/NICER/run1/config.ini --multinest --resume > out 2> err

mkdir $HOME/J0740_STU/modules/NICER/run1/init
mv out $HOME/J0740_STU/modules/NICER/run1/init
mv err $HOME/J0740_STU/modules/NICER/run1/init
mv samples $HOME/J0740_STU/modules/NICER/run1
#end of job file
