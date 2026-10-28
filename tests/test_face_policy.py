import importlib.util
from pathlib import Path
import pytest

path = Path("vendor/ComfyUI-PhotoMaker-Plus/face_policy.py")
spec = importlib.util.spec_from_file_location("face_policy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_single_face_is_selected():
    face = {"embedding": [0.1, 0.2]}
    assert module.select_single_face([face]) is face

@pytest.mark.parametrize("faces,code", [([], "NO_FACE_DETECTED"), ([{}, {}], "MULTIPLE_FACES_DETECTED")])
def test_invalid_face_count(faces, code):
    with pytest.raises(ValueError, match=code):
        module.select_single_face(faces)


def test_single_output_face_is_accepted():
    face = {"embedding": [0.3, 0.4]}
    assert module.select_single_output_face([face]) is face


@pytest.mark.parametrize(
    "faces,code",
    [([], "OUTPUT_NO_FACE_DETECTED"), ([{}, {}], "OUTPUT_MULTIPLE_FACES_DETECTED")],
)
def test_invalid_output_face_count_is_rejected(faces, code):
    with pytest.raises(ValueError, match=code):
        module.select_single_output_face(faces)
