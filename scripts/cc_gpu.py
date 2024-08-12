import argparse
import os
import subprocess

from automate import load_config, get_partition


EXEC_DIR = "$SCRATCH"  # Location to dispatch job and store results

# Resource allocation per GPU bundle: (CPUs, memory)
BUNDLE_DEFS = {
    'beluga': (10, '46G'),
    'cedar': (6, '32G'),
    'graham': (8, '32G'),
}


def main(account: str, time: str, gpus: int, bundle: str, job_name: str, go: bool):
    exec_dir = os.path.expandvars(EXEC_DIR)
    assert os.path.exists(exec_dir), f"{exec_dir} does not exist"

    # Get hardware bundle values
    cpus_per_gpu, mem_per_gpu = BUNDLE_DEFS[bundle]

    # Save current directory and then change to dispatch directory
    project_dir = os.getcwd()
    os.chdir(exec_dir)

    # Make sure output directories exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Set location of stdout/stderr log
    log_path = os.path.join(logs_dir, f"{job_name}_%A-%a.out")

    # Fill in Slurm script template
    slurm_script = f"""#!/bin/bash

#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_gpu}
#SBATCH --mem={mem_per_gpu}

job_id="$SLURM_JOB_ID ($SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT)"
echo "Starting job $job_id at `date`"

# Load virtual env first to avoid dependency errors
source {project_dir}/.venv/bin/activate
module load StdEnv/2023 gcc python scipy-stack
module load cuda/12.3

cmd="python {project_dir}/scripts/automate.py --save_dir {output_dir} --partition $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT --max_parallel {cpus_per_gpu} --go"
echo $cmd
$cmd

echo ""
echo "Completed job $job_id at `date`"
"""

    with open('slurm.sh', 'w') as f:
        f.write(slurm_script)

    if not go:
        print(f"Allocating {gpus} GPUs ({cpus_per_gpu} CPUs + {mem_per_gpu} memory each) for time '{time}'")
        data_files = load_config(output_dir, meta_seed=0)

        for i in range(gpus):
            partition = get_partition(data_files, f"{i+1}/{gpus}")
            num_trials = len([f for f in partition if not f.exists()])
            load_factor = round(num_trials / cpus_per_gpu, 2)
            print(f"  - GPU {i}: {num_trials} trials ({load_factor}x load factor)")

        print("*** This was just a test! The job was not dispatched to the scheduler.")
        print(f"*** If {exec_dir}/slurm.sh looks correct, re-run with the '--go' argument.")

    else:
        subprocess.run(f"sbatch --array=1-{gpus} slurm.sh", shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('account', type=str)
    parser.add_argument('time', type=str)
    parser.add_argument('gpus', type=int)
    parser.add_argument('bundle', type=str)
    parser.add_argument('--job_name', type=str, default='netraces')
    parser.add_argument('--go', action='store_true')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
