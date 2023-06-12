import argparse
from pathlib import Path
import subprocess

from simsiam import get_args_parser

if __name__ == "__main__":
    parser = get_args_parser()
    args, _ = parser.parse_known_args()

    slurm_parser = argparse.ArgumentParser("SlurmParser")
    slurm_parser.add_argument("--gpus", default=1, type=int)
    slurm_parser.add_argument("-p", "--partition", default="mlhiwidlc_gpu-rtx2080-advanced", type=str)
    slurm_parser.add_argument("--array", default=0, type=int)
    slurm_parser.add_argument("-t", "--time", default="23:59:59", type=str)
    slurm_parser.add_argument("--prefix", default="simsiam-argmin_sim", type=str)
    slurm_parser.add_argument("--suffix", default="default", type=str)
    slurm_args, _ = slurm_parser.parse_known_args()

    exp_dir = "/work/dlclarge2/rapanti-metassl-dino-stn/experiments"
    prefix = slurm_args.prefix
    suffix = slurm_args.suffix
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

    exp_name = f"{prefix}-{arch}-{dataset}-ep_{epochs}-bs_{bs}-seed_{seed}-{suffix}"
    args.output_dir = output_dir = Path(exp_dir).joinpath(exp_name)
    print(f"Experiment: {output_dir}")

    log_dir = output_dir.joinpath("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    code_dir = output_dir.joinpath("code")
    code_dir.mkdir(parents=True, exist_ok=True)
    copy_msg = subprocess.call(["cp", "-r", ".", code_dir])

    # Slurm settings
    log_file = log_dir.joinpath("%A.%a.%N.out")
    sbatch = [
        "#!/bin/bash", f"#SBATCH -p {slurm_args.partition}",
        f"#SBATCH -t {slurm_args.time}",
        f"#SBATCH --gres=gpu:{slurm_args.gpus}",
        f"#SBATCH -J {exp_name}",
        f"#SBATCH -o {log_file}",
        f"#SBATCH -e {log_file}",
        f"#SBATCH --array 0-{slurm_args.array}%1\n" if slurm_args.array > 0 else '',
        'echo "Workingdir: $PWD"',
        'echo "Started at $(date)"',
        'echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"\n',
        "source ~/.profile",
        "conda activate torch"
    ]
    run = [
        "torchrun",
        f"--nproc_per_node={slurm_args.gpus}",
        f"--nnodes=1", f"--standalone",
        f"code/run_train_eval.py"
    ]

    job_file = output_dir.joinpath("job.sh")
    with open(job_file, 'w') as file:
        for line in sbatch:
            file.write(line + " \n")
        file.write("\n")

        for line in run:
            file.write(line + " \\\n")

        for n, (key, value) in enumerate(sorted(dict(vars(args)).items())):
            if n:
                file.write(f" \\\n")
            file.write(f"--{key} {value}")
        file.write("\n")

    out = subprocess.call(["sbatch", job_file], cwd=output_dir)
