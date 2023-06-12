import argparse
from pathlib import Path
import subprocess

from simsiam import get_args_parser

if __name__ == "__main__":
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    slurm_parser = argparse.ArgumentParser("SlurmParser")
    slurm_parser.add_argument("--gpus", default=4, type=int)
    slurm_parser.add_argument("--array", default=3, type=int)
    slurm_parser.add_argument("--time", default="23:59:59", type=str)
    slurm_parser.add_argument("--prefix", default="simsiam", type=str)
    slurm_parser.add_argument("--suffix", default="def", type=str)
    slurm_args, _ = slurm_parser.parse_known_args()

    exp_dir = "/work/dlclarge2/rapanti-metassl-dino-stn/experiments"
    seed = args.seed
    arch = args.arch
    epochs = args.epochs
    bs = args.batch_size
    dataset = args.dataset
    match dataset:
        case "CIFAR10":
            args.data_path = "/work/dlclarge2/rapanti-metassl-dino-stn/datasets/CIFAR10"
        case "ImageNet":
            args.data_path = "/data/datasets/ImageNet/imagenet-pytorch"
        case _:
            args.data_path = "."

    exp_name = f"{slurm_args.prefix}-{arch}-{dataset}-ep_{epochs}-bs_{bs}-seed_{seed}-{slurm_args.suffix}"
    args.output_dir = output_dir = f"{exp_dir}/{exp_name}"

    exp_stg = dict(vars(args))

    log_dir = Path(output_dir).joinpath("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    copy_msg = subprocess.call(["cp", "-r", ".", output_dir])
    print(f"Experiment: {exp_name}")

    # Slurm settings
    partition = "mlhiwidlc_gpu-rtx2080-advanced"
    time = slurm_args.time
    gpus = slurm_args.gpus
    log_file = f"{exp_dir}/{exp_name}/log/%A.%a.%N.out"
    sbatch_array = slurm_args.array
    sbatch = [
        "#!/bin/bash", f"#SBATCH -p {partition}",
        f"#SBATCH -t {time}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH -J {exp_name}",
        f"#SBATCH -o {log_file}",
        f"#SBATCH -e {log_file}",
        f"#SBATCH --array 0-{sbatch_array}%1" if sbatch_array > 0 else '',
        'echo "Workingdir: $PWD"',
        'echo "Started at $(date)"',
        'echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"',
        "source ~/.profile",
        "source activate torch"
    ]
    run = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        f"--nnodes=1", f"--standalone",
        f"run_train_eval.py"
    ]
    job_file = Path(output_dir).joinpath("job.sh")
    with open(job_file, 'w') as file:
        for line in sbatch:
            file.write(line + " \n")
        file.write("\n")
        for line in run:
            file.write(line + " \\\n")
        for key, value in exp_stg.items():
            file.write(f"--{key} {value} ")
        file.write("\n")

    out = subprocess.call(["sbatch", job_file])
