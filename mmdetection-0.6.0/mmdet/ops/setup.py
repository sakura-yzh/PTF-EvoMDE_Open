import os
import subprocess
import sys

MODULES = ['dcn', 'nms', 'roi_align', 'roi_pool', 'sigmoid_focal_loss']

def build_module(path):
    if not os.path.isfile(os.path.join(path, 'setup.py')):
        print(f'[skip] No setup.py in {path}')
        return

    try:
        subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=path)
        print(f'[ok] Built {os.path.basename(path)}')
    except subprocess.CalledProcessError:
        print(f'[fail] Build failed: {os.path.basename(path)}')

def main():
    for mod in MODULES:
        build_module(mod)

if __name__ == '__main__':
    main()
