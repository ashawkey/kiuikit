import re
from importlib_metadata import version

# only partial support of the full requirements.txt grammar
# ref: https://pip.pypa.io/en/stable/reference/requirements-file-format/

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('requirements', type=str, help='path to requirements.txt')
    parser.add_argument('--out', type=str, default='requirements.lock.txt', help='output file path')
    opt = parser.parse_args()

    # read requirements.txt
    with open(opt.requirements, 'r') as f:
        requirements = f.readlines()
    
    print(f'[INFO] lock current versions for {len(requirements)} packages...')
    with open(opt.out, 'w') as f:
        for line in requirements:
            line = line.strip()
            # ignore
            if line.startswith('#') or line.startswith('-') or len(line) == 0:
                f.write(line + '\n')
                continue
            # parse version
            name = re.split(r'\s*[=><]=\s*', line)[0]
            # remove variant, e.g., name[full] --> name
            pure_name = re.split(r'\[.*\]', name)[0]
            ver = version(pure_name)
            f.write(f'{name} == {ver}\n')
            print(f'{name} == {ver}')
    print(f'[INFO] write locked versions to {opt.out}')