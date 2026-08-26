"""Command-line interface for localise.

Provides the `localise` command with subcommands covering the full pipeline:

    localise prepare-masks   ...   # anatomical masks in reference space
    localise prepare-tracts  ...   # probtrackx2 tract-density maps
    localise predict         ...   # localise a structure with a (pre)trained model
    localise train           ...   # train a model on high-quality labels
    localise connectivity-driven . # classic thresholded-overlap localisation
"""

import argparse
import sys
import textwrap

from localise.args import (
    add_predict_arguments, add_train_arguments,
    validate_predict, validate_train,
)


def _run_prepare_masks(args):
    from localise import prepare_masks
    prepare_masks.run(args)


def _run_prepare_tracts(args):
    from localise import prepare_tracts
    prepare_tracts.run(args)


def _run_connectivity_driven(args):
    from localise import connectivity_driven
    connectivity_driven.run(args)


def _run_predict(args):
    from localise.modes import predict_mode
    predict_mode(
        masks=args.masks, seed=args.seed, structure=args.structure,
        tracts=args.tracts, tracts_list=args.tracts_list, data=args.data,
        atlas=args.atlas, out=args.out, model=args.model,
        spatial=args.spatial, hemisphere=args.hemisphere, verbose=args.verbose
    )


def _run_train(args):
    from localise.modes import train_mode
    train_mode(
        masks=args.masks, labels=args.labels, tracts=args.tracts,
        tracts_list=args.tracts_list, seed=args.seed, data=args.data,
        atlas=args.atlas, out_model=args.out_model, spatial=args.spatial,
        hemisphere=args.hemisphere, epochs=args.epochs, verbose=args.verbose
    )


def build_parser():
    """Build the localise argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog='localise',
        description=textwrap.dedent("""
            Localise - A tool for localising brain structures using connectivity-based features.

            This tool can train models to localise specific brain structures (like VIM, LGN)
            using diffusion MRI tractography data, or use pre-trained models for prediction.
            A typical pipeline is:

                localise prepare-masks   --ref t1.nii.gz --aparc aparc.a2009s+aseg.nii.gz \\
                                         --warp std2native_warp.nii.gz --structure vim --out sub01
                localise prepare-tracts  --bpx sub01/dMRI.bedpostX --masks sub01/masks \\
                                         --structure vim --out sub01/streamlines
                localise predict         --masks sub01/masks --tracts sub01/streamlines \\
                                         --structure vim --spatial --out sub01
        """).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    from localise import prepare_masks, prepare_tracts, connectivity_driven

    p = subparsers.add_parser(
        'prepare-masks',
        help='Create anatomical masks in reference space (both hemispheres).',
        description=prepare_masks.DESCRIPTION, epilog=prepare_masks.EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    prepare_masks.add_arguments(p)
    p.set_defaults(func=_run_prepare_masks)

    p = subparsers.add_parser(
        'prepare-tracts',
        help='Run probtrackx2 tractography (both hemispheres by default).',
        description=prepare_tracts.DESCRIPTION, epilog=prepare_tracts.EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    prepare_tracts.add_arguments(p)
    p.set_defaults(func=_run_prepare_tracts)

    p = subparsers.add_parser(
        'predict',
        help='Localise a structure with a pre-trained or custom model '
             '(both hemispheres by default).',
        formatter_class=argparse.RawTextHelpFormatter
    )
    add_predict_arguments(p)
    p.set_defaults(func=_run_predict, validate=validate_predict)

    p = subparsers.add_parser(
        'train',
        help='Train a model to localise a structure.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    add_train_arguments(p)
    p.set_defaults(func=_run_train, validate=validate_train)

    p = subparsers.add_parser(
        'connectivity-driven',
        help='Combine thresholded tract-density maps into a binary mask.',
        description=connectivity_driven.DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    connectivity_driven.add_arguments(p)
    p.set_defaults(func=_run_connectivity_driven)

    return parser


def parse_args(argv=None):
    """Parse (and validate) command line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate = getattr(args, 'validate', None)
    if validate is not None:
        validate(parser, args)
    return args


def main(argv=None):
    """Main entry point for the localise command."""
    args = parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
