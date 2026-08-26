import pytest
from localise.modes import RESOURCES_PATH
from localise.modes import (
    check_values, check_prediction_params,
    _handle_default_model, check_training_params,
    train_mode, predict_mode
)
from pathlib import Path

path_to_data = Path(__file__).parent / 'test_data'

def test_check_values(tmp_path):
    masks = tmp_path / 'masks.txt'
    masks.write_text('sub1/left/probmap.nii.gz\nsub1/right/probmap.nii.gz')
    assert check_values(masks) == [
        Path('sub1/left/probmap.nii.gz'), 
        Path('sub1/right/probmap.nii.gz')
    ]
    masks = tmp_path / 'masks.nii.gz'
    assert check_values(masks) == [Path(tmp_path / 'masks.nii.gz')]
    tracts = tmp_path / 'tracts'
    assert check_values(tracts) == [Path(tmp_path / 'tracts')]

def test_check_prediction_params(tmp_path):
    # for a single subject
    masks = tmp_path / 'masks'
    tracts = tmp_path / 'tracts'
    structure = 'vim'
    out = tmp_path / 'out'
    data_type = 'single32'
    hemisphere = 'left'
    params = check_prediction_params(
        masks=masks, structure=structure, tracts=tracts, 
        out=out, data_type=data_type, hemisphere=hemisphere, 
        spatial=True
    )
    assert params['masks'] == [tmp_path / 'masks' / hemisphere]
    assert params['tracts'] == [tmp_path / 'tracts' / hemisphere]
    assert params['structure'] == 'vim'
    assert params['seed'] == [masks / hemisphere / 'tha.nii.gz']
    assert params['tracts_list'] == (RESOURCES_PATH / 'data' / 
                                     f'{structure}_default_target_list.txt')
    assert params['atlas'] == [None]
    assert params['out'] == [out / hemisphere / 'probmap.nii.gz']
    assert params['hemisphere'] == 'left'
    assert params['spatial']
    
    # for a group of subjects in txt
    masks = tmp_path / 'masks.txt'
    masks.write_text('sub1/masks\nsub2/masks')
    tracts = tmp_path / 'tracts.txt'
    tracts.write_text('sub1/tracts/\nsub2/tracts/')
    out = tmp_path / 'out.txt'
    out.write_text('sub1/out\nsub2/out')
    params = check_prediction_params(
        masks=masks, structure=structure, tracts=tracts, 
        out=out, data_type=data_type, hemisphere=hemisphere, 
        spatial=True
    )
    assert params['masks'] == [
        Path('sub1/masks') / hemisphere, Path('sub2/masks') / hemisphere
    ]
    assert params['tracts'] == [
        Path(f'sub1/tracts/{hemisphere}'), Path(f'sub2/tracts/{hemisphere}')
    ]
    assert params['out'] == [
        Path(f'sub1/out/{hemisphere}/probmap.nii.gz'), 
        Path(f'sub2/out/{hemisphere}/probmap.nii.gz')
    ]
    assert params['model'] == (RESOURCES_PATH / 'models' / structure / data_type / 
                               hemisphere / 'spatial_model.pth')
    
    # other scenarios
    model = tmp_path / 'model.pth'
    seed = tmp_path / 'seeds.txt'
    seed.write_text('sub1/left/seeds.nii.gz\nsub2/left/seeds.nii.gz')
    tracts_list = tmp_path / 'tracts_list.txt'
    params = check_prediction_params(
        masks=masks, model=model, seed=seed, tracts=tracts, 
        out=out, data_type=data_type, hemisphere=hemisphere, 
        tracts_list=tracts_list, spatial=True
    )
    assert params['masks'] == [
        Path('sub1/masks') / hemisphere, Path('sub2/masks') / hemisphere
    ]
    assert params['tracts'] == [
        Path(f'sub1/tracts/{hemisphere}'), Path(f'sub2/tracts/{hemisphere}')
    ]
    assert params['out'] == [
        Path(f'sub1/out/{hemisphere}/probmap.nii.gz'), 
        Path(f'sub2/out/{hemisphere}/probmap.nii.gz')
    ]
    assert params['seed'] == [
        Path('sub1/left/seeds.nii.gz'), Path('sub2/left/seeds.nii.gz')
    ]
    assert params['model'] == model
    
    data = tmp_path / 'data.txt'
    data.write_text('sub1/data.npy\nsub2/data.npy')
    params = check_prediction_params(
        masks=masks, data=data, model=model, seed=seed,
        out=out, data_type=data_type, hemisphere=hemisphere, 
        spatial=True, 
    )
    assert params['data'] == [Path('sub1/data.npy'), Path('sub2/data.npy')]

def test_handle_default_model():
    assert _handle_default_model('vim', 'single32', 'left', True, None) == (
        RESOURCES_PATH / 'models' / 'vim' / 'single32' / 'left' / 
        'spatial_model.pth'
    )
    assert _handle_default_model('vim', 'single32', 'left', False, 'atlas.nii.gz') == (
        RESOURCES_PATH / 'models' / 'vim' / 'single32' / 'left' / 
        'model_with_prior.pth'
    )
    
def test_check_training_params(tmp_path):
    # for multiple subjects
    seed = tmp_path / 'seeds.txt'
    seed.write_text('sub1/left/seeds.nii.gz\nsub2/left/seeds.nii.gz')
    labels = tmp_path / 'labels.txt'
    labels.write_text('sub1/left/labels.nii.gz\nsub2/left/labels.nii.gz')
    masks = tmp_path / 'masks.txt'
    masks.write_text('sub1/masks\nsub2/masks')
    tracts = tmp_path / 'tracts.txt'
    tracts.write_text('sub1/tracts/\nsub2/tracts/')
    hemisphere = 'left'
    out_model = tmp_path / 'out_model.pth'
    tracts_list = tmp_path / 'tracts_list.txt'
    params = check_training_params(
        masks=masks, labels=labels, seed=seed, tracts=tracts, 
        tracts_list=tracts_list, hemisphere=hemisphere, out_model=out_model
    )
    assert params['masks'] == [
        Path('sub1/masks') / hemisphere, Path('sub2/masks') / hemisphere
    ]
    assert params['labels'] == [
        Path('sub1/left/labels.nii.gz'), Path('sub2/left/labels.nii.gz')
    ]
    assert params['seed'] == [
        Path('sub1/left/seeds.nii.gz'), Path('sub2/left/seeds.nii.gz')
    ]
    assert params['tracts'] == [
        Path(f'sub1/tracts/{hemisphere}'), Path(f'sub2/tracts/{hemisphere}')
    ]
    assert params['hemisphere'] == 'left'
    assert params['tracts_list'] == tmp_path / 'tracts_list.txt'
    
    # for a single subject
    seed = tmp_path / 'seeds.nii.gz'
    labels = tmp_path / 'labels.nii.gz'
    tracts = tmp_path / 'tracts'
    hemisphere = 'right'
    params = check_training_params(
        labels=labels, seed=seed, tracts=tracts, 
        tracts_list=tracts_list, 
        hemisphere=hemisphere, out_model=out_model
    )
    assert params['masks'] == [None]
    assert params['labels'] == [tmp_path / 'labels.nii.gz']
    assert params['seed'] == [tmp_path / 'seeds.nii.gz']
    assert params['tracts'] == [tmp_path / 'tracts' / 'right']
    assert params['hemisphere'] == 'right'
    assert params['out_model'] == tmp_path / 'out_model.pth'
    
def test_train_and_predict_round_trip(tmp_path):
    # train a small custom model on the bundled test subject, then use it
    # in prediction mode: this exercises the full train_mode/predict_mode glue
    sub = path_to_data / '100610'
    seed = sub / 'roi' / 'left' / 'tha_small.nii.gz'
    labels = sub / 'high-quality-labels' / 'left' / 'labels.nii.gz'
    tracts = sub / 'streamlines'  # hemisphere is appended by the param layer
    tracts_list = tmp_path / 'tracts_list.txt'
    tracts_list.write_text('seeds_to_11101_1.nii.gz\nseeds_to_11102_1.nii.gz\n')
    out_model = tmp_path / 'model.pth'

    model = train_mode(masks=None, labels=str(labels), tracts=str(tracts),
                       tracts_list=str(tracts_list), seed=str(seed),
                       out_model=str(out_model), spatial=True,
                       hemisphere='left', epochs=2, verbose=False)
    assert out_model.exists()

    out = tmp_path / 'out'
    predictions = predict_mode(masks=None, seed=str(seed), tracts=str(tracts),
                               tracts_list=str(tracts_list), out=str(out),
                               model=str(out_model), spatial=True,
                               hemisphere='left', verbose=False)
    assert len(predictions) == 1
    assert (out / 'left' / 'probmap.nii.gz').exists()


def test_predict_mode_requires_hemisphere_with_explicit_paths(tmp_path):
    # both-hemisphere mode (hemisphere=None) is only valid with the standard
    # folder layout; explicit seed/data/model paths are hemisphere-specific
    with pytest.raises(ValueError, match='hemisphere'):
        predict_mode(masks=None, seed='seed.nii.gz', tracts='tracts',
                     tracts_list='list.txt', out=str(tmp_path),
                     model='model.pth', hemisphere=None, verbose=False)


def test_train_mode_requires_hemisphere_with_tracts(tmp_path):
    with pytest.raises(ValueError, match='hemisphere'):
        train_mode(masks=None, labels='labels.nii.gz', tracts='tracts',
                   tracts_list='list.txt', seed='seed.nii.gz',
                   out_model=str(tmp_path / 'model.pth'),
                   hemisphere=None, verbose=False)


def test_check_prediction_params_detects_tracts_list(tmp_path):
    # a tracts_list.txt generated by prepare-tracts next to the tract maps
    # takes precedence over the shipped default list
    masks = tmp_path / 'masks'
    tracts = tmp_path / 'tracts'
    (tracts / 'left').mkdir(parents=True)
    detected = tracts / 'left' / 'tracts_list.txt'
    detected.write_text('seeds_to_1.nii.gz\nseeds_to_2.nii.gz\n')

    params = check_prediction_params(
        masks=masks, structure='vim', tracts=tracts, out=tmp_path / 'out',
        data_type='single32', hemisphere='left', spatial=True
    )
    assert params['tracts_list'] == detected


def test_predict_mode_with_shipped_default_model(tmp_path):
    # exercise the default-model path end to end: pre-saved features with the
    # canonical 160 targets, the shipped vim 2mm spatial model, real output
    import numpy as np
    import nibabel as nib

    seed = path_to_data / '100610' / 'roi' / 'left' / 'tha_small.nii.gz'
    n_voxels = int((nib.load(str(seed)).get_fdata() > 0).sum())
    rng = np.random.default_rng(0)
    data = tmp_path / 'features.npy'
    np.save(data, rng.random((160, n_voxels)).astype(np.float32))

    out = tmp_path / 'out'
    predictions = predict_mode(masks=None, seed=str(seed), data=str(data),
                               structure='vim', data_type='2mm', spatial=True,
                               out=str(out), hemisphere='left', verbose=False)

    assert len(predictions) == 1
    assert predictions[0].shape[0] == n_voxels
    assert (out / 'left' / 'probmap.nii.gz').exists()


def test_handle_default_model_missing(tmp_path):
    # unknown data type: the error should list what is actually available
    with pytest.raises(ValueError, match='2mm'):
        _handle_default_model('vim', 'nonexistent', 'left', True, None)
    # unshipped structure: the error should point at custom training
    with pytest.raises(ValueError, match='train your own'):
        _handle_default_model('lgn', 'single32', 'left', True, None)
