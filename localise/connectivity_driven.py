#!/usr/bin/env python3
"""
Connectivity-driven localisation.

Thresholds two (optionally three) tract-density maps, multiplies them, and
thresholds the product into a binary mask - the classic connectivity-driven
approach to targeting (e.g., M1 x contralateral cerebellum for VIM).

Copyright (C) 2019 University of Oxford
Written by Ying-Qiu Zheng (Python rewrite)
"""

import argparse
import sys
from pathlib import Path
from localise.utils import run_fsl_command, check_fsl_environment, fsl_bin


DESCRIPTION = (
    "Create a connectivity-driven mask by thresholding and multiplying "
    "tract-density maps (e.g., M1 and contralateral cerebellum for VIM)."
)


def add_arguments(parser):
    """Add connectivity-driven arguments to an argparse parser."""
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--target1', required=True,
        help='Target1 tract density map, e.g., M1'
    )
    required.add_argument(
        '--target2', required=True,
        help='Target2 tract density map, e.g., contralateral cerebellum'
    )
    required.add_argument(
        '--out', required=True,
        help='The output filename'
    )
    parser.add_argument(
        '--target3',
        help='Optional, target3 tract density map'
    )
    parser.add_argument(
        '--thr1', type=float, default=50,
        help='Threshold (percentile) on target1 tract density (default: %(default)s)'
    )
    parser.add_argument(
        '--thr2', type=float, default=50,
        help='Threshold (percentile) on target2 tract density (default: %(default)s)'
    )
    parser.add_argument(
        '--thr3', type=float, default=50,
        help='Threshold (percentile) on target3 tract density (default: %(default)s)'
    )
    parser.add_argument(
        '--thr', type=float, default=70,
        help='Final threshold (percentile) applied on the product map (default: %(default)s)'
    )
    return parser


def connectivity_driven(target1, target2, out, target3=None,
                        thr1=50, thr2=50, thr3=50, thr=70):
    """Threshold and multiply tract-density maps into a binary mask."""
    check_fsl_environment()

    targets = [target1, target2] + ([target3] if target3 else [])
    for target in targets:
        if not Path(target).exists():
            raise FileNotFoundError(f"{target} does not exist.")

    out_path = Path(out)
    out_dir = out_path.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [thr1, thr2, thr3][:len(targets)]
    thresholded = []
    for i, (target, t) in enumerate(zip(targets, thresholds), start=1):
        thresholded_map = out_dir / f"target{i}_thrP{t:g}"
        run_fsl_command([
            fsl_bin('fslmaths'), str(target), '-thrP', str(t), str(thresholded_map)
        ])
        thresholded.append(str(thresholded_map))

    cmd = [fsl_bin('fslmaths'), thresholded[0]]
    for thresholded_map in thresholded[1:]:
        cmd.extend(['-mul', thresholded_map])
    cmd.extend(['-thrP', str(thr), '-bin', str(out_path)])
    run_fsl_command(cmd)

    print(f"Connectivity-driven mask saved to: {out_path}")


def run(args):
    """Run connectivity-driven localisation from parsed arguments."""
    connectivity_driven(
        target1=args.target1, target2=args.target2, out=args.out,
        target3=args.target3, thr1=args.thr1, thr2=args.thr2,
        thr3=args.thr3, thr=args.thr
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_arguments(parser)
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        run(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
