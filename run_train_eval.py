import argparse
import time
from pathlib import Path

import eval_linear
import simsiam


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO TRAINING PIPELINE', parents=[simsiam.get_args_parser()])
    parser.add_argument("--pipeline_mode", default=('pretrain', 'eval'), choices=['pretrain', 'eval'], type=str, nargs='+')
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'pretrain' in args.pipeline_mode:
        print('STARTING PRETRAINING')
        simsiam.main(args)
        print('FINISHED PRETRAINING')

    time.sleep(10)

    if 'eval' in args.pipeline_mode:
        # change linear specific parameters
        args.epochs = 90
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 0
        args.batch_size = 4096
        args.lars = True
        args.pretrained = f"{args.output_dir}/checkpoint.pth"
        args.ckp_key = "model"
        args.val_freq = 1
        print('STARTING EVALUATION')
        eval_linear.main(args)
        print('FINISHED EVALUATION')
