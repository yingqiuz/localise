"""End-to-end integration tests for the pipeline, without FSL.

A fake FSLDIR is populated with stub executables that create real NIfTI output
files (copies of small templates) where their real counterparts would.
prepare-masks and prepare-tracts are then run for real (no mocks), exercising
command-file generation, $FSLDIR expansion, local execution via bash, fsl_sub
queue submission, and the contract that every file listed in the generated
tracts_list.txt is actually produced - which in turn lets `localise predict`
run on the prepared outputs with a real shipped model.
"""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest


FSLMATHS_STUB = """#!/usr/bin/env bash
# fake fslmaths: the last argument is the output image
out="${@: -1}"
case "$out" in *.nii.gz) : ;; *) out="$out.nii.gz" ;; esac
mkdir -p "$(dirname "$out")"
cp "__MASK__" "$out"
"""

APPLYWARP_STUB = """#!/usr/bin/env bash
# fake applywarp: output follows -o
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$out" ] || exit 1
case "$out" in *.nii.gz) : ;; *) out="$out.nii.gz" ;; esac
mkdir -p "$(dirname "$out")"
cp "__MASK__" "$out"
"""

FLIRT_STUB = """#!/usr/bin/env bash
# fake flirt: output follows -out
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$out" ] || exit 1
case "$out" in *.nii.gz) : ;; *) out="$out.nii.gz" ;; esac
mkdir -p "$(dirname "$out")"
cp "__MASK__" "$out"
"""

PROBTRACKX_STUB = """#!/usr/bin/env bash
# fake probtrackx2: creates seeds_to_<target> maps for every target listed
dir=""
targets=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) dir="$2"; shift 2 ;;
    --targetmasks) targets="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$dir" ] && [ -n "$targets" ] || exit 1
mkdir -p "$dir"
for t in $(cat "$targets"); do
  name="$(basename "$t")"
  name="${name%.nii.gz}"
  name="${name%.nii}"
  cp "__DATA__" "$dir/seeds_to_${name}.nii.gz"
done
cp "__DATA__" "$dir/fdt_paths.nii.gz"
"""

FSL_SUB_STUB = """#!/usr/bin/env bash
# fake fsl_sub: records its invocation, then runs the job synchronously
printf '%s\\n' "$*" >> "$FSLSUB_LOG"
task=""
cmd=()
while [ $# -gt 0 ]; do
  case "$1" in
    --has_queues) echo "Yes"; exit 0 ;;
    -N|-l|-j|--coprocessor) shift 2 ;;
    -n) shift ;;
    -t) task="$2"; shift 2 ;;
    *) cmd+=("$1"); shift ;;
  esac
done
if [ -n "$task" ]; then
  bash "$task" 1>&2
else
  "${cmd[@]}" 1>&2
fi
echo "job_1234"
"""


def _write_stub(path, content, mask=None, data=None):
    content = content.replace('__MASK__', str(mask or ''))
    content = content.replace('__DATA__', str(data or ''))
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_fsl(tmp_path, monkeypatch):
    """A fake FSLDIR whose binaries create real (template) NIfTI files."""
    import numpy as np
    import nibabel as nib

    fsldir = tmp_path / 'fakefsl'
    bindir = fsldir / 'bin'
    bindir.mkdir(parents=True)
    standard = fsldir / 'data' / 'standard'
    standard.mkdir(parents=True)

    # small real volumes for the stubs to copy: an all-ones binary mask, and
    # a random-valued map standing in for tract densities
    mask_t = fsldir / 'template_mask.nii.gz'
    data_t = fsldir / 'template_data.nii.gz'
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), affine), mask_t)
    rng = np.random.default_rng(0)
    nib.save(nib.Nifti1Image(rng.random((5, 5, 5)).astype(np.float32) * 100, affine), data_t)

    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), affine),
             standard / 'MNI152_T1_1mm_brain_mask.nii.gz')

    _write_stub(bindir / 'fslmaths', FSLMATHS_STUB, mask=mask_t)
    _write_stub(bindir / 'applywarp', APPLYWARP_STUB, mask=mask_t)
    _write_stub(bindir / 'flirt', FLIRT_STUB, mask=mask_t)
    _write_stub(bindir / 'probtrackx2', PROBTRACKX_STUB, data=data_t)
    _write_stub(bindir / 'probtrackx2_gpu', PROBTRACKX_STUB, data=data_t)

    monkeypatch.setenv('FSLDIR', str(fsldir))
    return fsldir


def _make_subject(tmp_path):
    sub = tmp_path / 'sub01'
    sub.mkdir()
    for name in ('t1.nii.gz', 'warp.nii.gz', 'aparc.nii.gz'):
        (sub / name).write_text('fake')
    return sub


def _make_bpx(sub):
    bpx = sub / 'dMRI.bedpostX'
    bpx.mkdir()
    (bpx / 'nodif_brain_mask.nii.gz').write_text('fake')
    return bpx


def _run_pipeline(sub, gpu=False):
    from localise.prepare_masks import create_masks
    from localise import prepare_tracts

    create_masks(ref=str(sub / 't1.nii.gz'), warp=str(sub / 'warp.nii.gz'),
                 out=str(sub), aparc=str(sub / 'aparc.nii.gz'), structure='vim')

    bpx = _make_bpx(sub)
    args = SimpleNamespace(bpx=str(bpx), masks=str(sub / 'masks'),
                           out=str(sub / 'streamlines'), structure='vim',
                           seed=None, hemisphere=None, warp=None, ref=None,
                           ptx_opts=None, gpu=gpu)
    prepare_tracts.run(args)


def _assert_pipeline_outputs(sub):
    masks = sub / 'masks'
    for hemi in ('left', 'right'):
        for name in ('1', '75', 'tha', 'vim', 'cerebellum', 'cerebellum_target',
                     'exclusion', 'scpct_mask1', 'scpct_mask15', 'stop_for_scpct'):
            assert (masks / hemi / f'{name}.nii.gz').exists(), f'{hemi}/{name}'
    assert (masks / 'cortex.nii.gz').exists()
    assert (masks / 'csf.nii.gz').exists()

    # the key contract: every file predict will look for was produced
    for hemi in ('left', 'right'):
        hemi_dir = sub / 'streamlines' / hemi
        tracts_list = hemi_dir / 'tracts_list.txt'
        assert tracts_list.exists()
        listed = tracts_list.read_text().splitlines()
        assert len(listed) == 160
        missing = [f for f in listed if not (hemi_dir / f).exists()]
        assert not missing, f'{hemi}: tract maps missing for {missing[:5]}'


def test_prepare_pipeline_local(fake_fsl, tmp_path, monkeypatch):
    """prepare-masks + prepare-tracts end to end, local (no queue) execution."""
    monkeypatch.setattr('localise.prepare_masks.check_fsl_sub_queues', lambda: False)
    monkeypatch.setattr('localise.prepare_tracts.check_fsl_sub_queues', lambda: False)

    sub = _make_subject(tmp_path)
    _run_pipeline(sub)
    _assert_pipeline_outputs(sub)


def test_prepare_pipeline_queue(fake_fsl, tmp_path, monkeypatch):
    """Same pipeline via fsl_sub submission (stub queue runs jobs in place)."""
    log = tmp_path / 'fsl_sub.log'
    log.write_text('')
    _write_stub(fake_fsl / 'bin' / 'fsl_sub', FSL_SUB_STUB)
    monkeypatch.setenv('FSLSUB_LOG', str(log))
    monkeypatch.setenv('PATH', f"{fake_fsl / 'bin'}{os.pathsep}{os.environ['PATH']}")

    sub = _make_subject(tmp_path)
    _run_pipeline(sub, gpu=True)
    _assert_pipeline_outputs(sub)

    entries = [e for e in log.read_text().splitlines()
               if '--has_queues' not in e]

    # the independent mask commands go in as a task array
    array_jobs = [e for e in entries if '-t' in e.split()]
    assert any('create_masks_cmds' in e for e in array_jobs)

    # the sequential SCPCT chain is one job (bash <file>), dependent on the array
    scpct_jobs = [e for e in entries if 'create_masks_scpct_cmds' in e]
    assert scpct_jobs
    for e in scpct_jobs:
        tokens = e.split()
        assert 'bash' in tokens and '-t' not in tokens and '-j' in tokens

    # two probtrackx runs per hemisphere; the SCPCT run waits for the first,
    # and gpu submission requests the cuda coprocessor
    ptx_jobs = [e for e in entries if 'probtrackx2' in e]
    assert len(ptx_jobs) == 4
    assert sum('-j' in e.split() for e in ptx_jobs) == 2
    assert all('--coprocessor' in e.split() for e in ptx_jobs)


def test_full_pipeline_through_predict(fake_fsl, tmp_path, monkeypatch):
    """The complete user journey: prepare-masks -> prepare-tracts ->
    localise predict with the real shipped model, both hemispheres."""
    import nibabel as nib
    import numpy as np
    from localise.cli import main

    monkeypatch.setattr('localise.prepare_masks.check_fsl_sub_queues', lambda: False)
    monkeypatch.setattr('localise.prepare_tracts.check_fsl_sub_queues', lambda: False)

    sub = _make_subject(tmp_path)
    _run_pipeline(sub)
    _assert_pipeline_outputs(sub)

    out = sub / 'localised'
    main(['predict', '--masks', str(sub / 'masks'),
          '--tracts', str(sub / 'streamlines'),
          '--structure', 'vim', '--spatial', '--out', str(out)])

    for hemi in ('left', 'right'):
        probmap_file = out / hemi / 'probmap.nii.gz'
        assert probmap_file.exists(), f'{hemi} probmap missing'
        probmap = nib.load(str(probmap_file)).get_fdata()
        assert probmap.shape == (5, 5, 5)
        assert np.all((probmap >= 0) & (probmap <= 1))
