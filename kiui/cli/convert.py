from kiui.mesh import Mesh

if __name__ == '__main__':
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help="path to input mesh")
    parser.add_argument('out', type=str, help="path to output mesh")

    opt = parser.parse_args()

    mesh = Mesh.load(opt.mesh)
    mesh.write(opt.out)