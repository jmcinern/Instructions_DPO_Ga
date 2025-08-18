#!/bin/bash
#SBATCH --job-name=test_gen_IRT
#SBATCH --output=./out/IRT.GEN%j.out
#SBATCH --error=./err/IRT_GEN%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=k2-lowpri  # Changed from k2-gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G  # Reduced from 256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

# Load only required modules (no GPU modules needed)
module load python3/3.10.5/gcc-9.3.0
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
pip install -r requirements.txt

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

python Create_Model_Comparison.py