import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


def _write_payload(path: Path, eeg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, {"preprocessed_eeg_data": eeg})


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    from nvol_retrieval import data

    monkeypatch.setattr(
        data,
        "EXPECTED_SHAPES",
        {"train": (2, 4, 3, 5), "test": (2, 80, 3, 5)},
    )
    root = tmp_path / "Things_eeg"
    train = np.arange(2 * 4 * 3 * 5, dtype=np.float64).reshape(2, 4, 3, 5)
    test = np.arange(2 * 80 * 3 * 5, dtype=np.float64).reshape(2, 80, 3, 5)
    _write_payload(root / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy", train)
    _write_payload(root / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy", test)
    paths = root / "image_set/img_path"
    paths.mkdir(parents=True)
    np.save(paths / "train_image.npy", np.array(["/x/a.jpg", "/x/a.jpg", "/x/b.jpg", "/x/b.jpg"]))
    np.save(paths / "test_image.npy", np.array(["/x/c.jpg", "/x/c.jpg", "/x/d.jpg", "/x/d.jpg"]))
    return root


def test_sub08_contract_preserves_train_repeats_and_ordered_test_names(data_root):
    from nvol_retrieval.data import load_sub08_split

    train = load_sub08_split(data_root, "train")
    test = load_sub08_split(data_root, "test")
    assert train.eeg.shape == (8, 3, 5)
    assert train.image_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert train.repeats == 4
    assert test.eeg.shape == (2, 80, 3, 5)
    assert test.names == ["c", "d"]
    assert test.image_idx.tolist() == [0, 1]
    assert test.repeats == 80


def test_only_sub08_is_accepted(data_root):
    from nvol_retrieval.data import load_sub08_split

    with pytest.raises(ValueError, match="sub-08"):
        load_sub08_split(data_root, "train", subject=7)


def test_rejects_malformed_payload_shape_and_caption_duplication(data_root):
    from nvol_retrieval.data import load_sub08_split

    bad = data_root / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
    np.save(bad, {"wrong_key": np.zeros((2, 4, 3, 5), dtype=np.float32)})
    with pytest.raises(ValueError, match="preprocessed_eeg_data"):
        load_sub08_split(data_root, "train")

    _write_payload(bad, np.zeros((2, 3, 3, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        load_sub08_split(data_root, "train")

    _write_payload(bad, np.zeros((2, 4, 3, 5), dtype=np.float64))
    np.save(data_root / "image_set/img_path/train_image.npy", np.array(["/x/a.jpg", "/x/b.jpg", "/x/b.jpg", "/x/b.jpg"]))
    with pytest.raises(ValueError, match="captions"):
        load_sub08_split(data_root, "train")


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_rejects_nonofficial_eeg_dtypes(data_root, dtype):
    from nvol_retrieval.data import load_sub08_split

    _write_payload(
        data_root / "Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy",
        np.zeros((2, 4, 3, 5), dtype=dtype),
    )
    with pytest.raises(ValueError, match="float64"):
        load_sub08_split(data_root, "train")


def test_inspector_refuses_to_write_under_raw_data_root(data_root):
    from inspect_sub08_data import _safe_output

    with pytest.raises(ValueError, match="outside --data-root"):
        _safe_output(data_root, data_root / "reports" / "data_contract.json")
