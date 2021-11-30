#!/bin/bash

job_directory=$PWD/jobs
mkdir -p $job_directory

for i in {0..5..1}
do
    job_file="${job_directory}/run_$i.job"

    echo "#!/bin/bash
        #SBATCH --job-name=NN_data_generation
        #SBATCH --partition=cmt
        #SBATCH --nodes=1
        #SBATCH --tasks-per-node=1
        #SBATCH --cpus-per-task=40
        #SBATCH --threads-per-core=1
        #SBATCH --time=7-00:00:00
        #SBATCH --mem=100Gb
        #SBATCH --output=../slurm_output/job-%j.out
        #SBATCH --error=../slurm_output/job-%j.err

        python generate_NN_data.py $i 20
        "
    sbatch $job_file
done




