import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--model', choices=['sd14', 'wd12', 'wd13', 'trinart', 'nai', 'nai-sfw'], default='sd14')
    args, additional = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.model == 'sd14':
        additional += '--ckpt ./sd_models/sd-v1-4.ckpt'.split()
    elif args.model == 'wd12':
        additional += '--ckpt ./sd_models/wd-v1-2-full-ema.ckpt'.split()
    elif args.model == 'wd13':
        additional += '--ckpt ./sd_models/wd-v1-3-float32.ckpt'.split()
    elif args.model == 'trinart':
        additional += '--ckpt ./sd_models/trinart2_step115000.ckpt'.split()
    elif args.model == 'nai':
        additional += '--ckpt ./sd_models/animefull-final-pruned.ckpt'.split()
    elif args.model == 'nai-sfw':
        additional += '--ckpt ./sd_models/animesfw-final-pruned.ckpt'.split()
    else:
        raise NotImplementedError

    # print('python webui.py ' + ' '.join(additional))
    p = subprocess.Popen('python webui.py ' + ' '.join(additional), shell=True)
    p.wait()


if __name__ == '__main__':
    main()
