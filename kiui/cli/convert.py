import os
import glob
import argparse
from kiui.mesh import Mesh

def process(fin, fout):
    if fout.endswith('.mp4'):
        os.system(f'kire {fin} --save_video {fout} --elevation -15 --wogui')
    else:
        mesh = Mesh.load(fin)
        mesh.write(fout)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inp', type=str, help="path to input (file or folder)")
    parser.add_argument('out', type=str, help="path to output (file or folder)")
    parser.add_argument('--inp_fmt', type=str, default=None, help="if inp is a folder, specify the input format")
    parser.add_argument('--out_fmt', type=str, default=None, help="if out is a folder, specify the output format")

    opt = parser.parse_args()

    if os.path.isdir(opt.inp):
        assert opt.out_fmt is not None, "if inp is a folder, fmt must be specified"
        os.makedirs(opt.out, exist_ok=True)
        for f in glob.glob(os.path.join(opt.inp, '*')):
            if opt.inp_fmt is not None and not f.endswith(opt.inp_fmt):
                continue
            fout = os.path.join(opt.out, os.path.basename(f).split('.')[0] + opt.out_fmt)
            process(f, fout)
    else:
        process(opt.inp, opt.out)
        


if __name__ == '__main__':
    main()