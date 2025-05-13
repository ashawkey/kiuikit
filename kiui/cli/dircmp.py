#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import os

import rich
from rich.console import Console

import filecmp
import difflib


def report_difference(dcmp):

    console = Console()

    console.rule(f'[green]{dcmp.left}[/green] v.s. [blue]{dcmp.right}[/blue]', style='white')

    # common
    commons = dcmp.common_files
    if len(commons) != 0:
        console.print(f'[Common] ' + ', '.join(commons))
    
    # only
    left_only = dcmp.left_only
    if len(left_only) != 0:
        console.print(f'[Left Only] ' + ', '.join(left_only), style='green')

    right_only = dcmp.right_only
    if len(right_only) != 0:
        console.print(f'[Right Only] ' + ', '.join(right_only), style='blue')
    
    # diff
    for name in dcmp.diff_files:
        console.print(f'[Different] {name}', style='red')
        if any([name.endswith(s) for s in args.suffix]):
            with open(os.path.join(dcmp.left, name), 'r') as f1:
                with open(os.path.join(dcmp.right, name), 'r') as f2:
                    lines = difflib.unified_diff(f1.readlines(), f2.readlines())
                    lines = list(lines)[2:] # remove header
                    for line in lines:
                        if line[0] == '+':
                            console.print(line[:-1], style='green')
                        elif line[0] == '-':
                            console.print(line[:-1], style='blue')
                        elif line[0] == '@':
                            console.print(line[:-1], style='red')
                        else:
                            console.print(line[:-1])

    # recursive call
    for sub_dcmp in dcmp.subdirs.values():
        report_difference(sub_dcmp)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str, help='the first directory path')
    parser.add_argument('dir2', type=str, help='the second directory path')
    parser.add_argument('--ignore', type=str, nargs='*', help='patterns to ignore')
    parser.add_argument('--suffix', type=str, nargs='*', default=['.py', '.sh', '.c', '.cpp', '.cu', '.h', '.hpp', '.cc'], help='suffixes to further display file difference')

    args = parser.parse_args()
    dcmp = filecmp.dircmp(args.dir1, args.dir2, ignore=args.ignore)
    report_difference(dcmp)