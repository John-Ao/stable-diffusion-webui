import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='')
    args, additional = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # print('python webui.py ' + ' '.join(additional))
    p = subprocess.Popen('python webui.py ' + ' '.join(additional), shell=True)
    p.wait()


if __name__ == '__main__':
    main()
