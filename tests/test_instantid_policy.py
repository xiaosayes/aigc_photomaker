# Server-pending: InstantID face-count policy tests without ComfyUI execution.
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


package = types.ModuleType("instantid_vendor")
package.__path__ = [str(Path(__file__).parents[1] / "vendor" / "ComfyUI_InstantID")]
sys.modules.setdefault("instantid_vendor", package)
path = Path(__file__).parents[1] / "vendor" / "ComfyUI_InstantID" / "face_policy.py"
spec = importlib.util.spec_from_file_location("instantid_vendor.face_policy", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_single_input_and_output_face_are_accepted():
    face = {"embedding": [0.1, 0.2]}
    assert module.validate_face_count([face]) is face
    assert module.validate_face_count([face], output=True) is face


@pytest.mark.parametrize(
    ("faces", "output", "code"),
    [
        ([], False, "NO_FACE_DETECTED"),
        ([{}, {}], False, "MULTIPLE_FACES_DETECTED"),
        ([], True, "OUTPUT_NO_FACE_DETECTED"),
        ([{}, {}], True, "OUTPUT_MULTIPLE_FACES_DETECTED"),
    ],
)
def test_invalid_face_count_is_rejected(faces, output, code):
    with pytest.raises(ValueError, match=code):
        module.validate_face_count(faces, output=output)


def test_registered_guards_convert_tensor_to_bgr_and_accept_one_face():
    face = {"embedding": [0.1]}

    class FakeAnalysis:
        def __init__(self):
            self.det_model = types.SimpleNamespace(input_size=None)
            self.images = []

        def get(self, image):
            self.images.append(image.copy())
            return [face]

    analysis = FakeAnalysis()
    images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    images[..., 0] = 1.0
    input_guard = module.InstantIDSingleFaceInputGuard()
    output_guard = module.InstantIDSingleFaceOutputGuard()

    assert input_guard.validate_input(images, analysis)[0] is images
    assert output_guard.validate_output(images, analysis)[0] is images
    assert analysis.images[0][0, 0, :].tolist() == [0, 0, 255]


def test_analyze_faces_retries_until_a_face_is_found():
    face = {"embedding": [0.1]}

    class FakeAnalysis:
        def __init__(self):
            self.det_model = types.SimpleNamespace(input_size=None)
            self.sizes = []

        def get(self, image):
            self.sizes.append(self.det_model.input_size)
            return [face] if self.det_model.input_size == (320, 320) else []

    analysis = FakeAnalysis()
    import numpy as np

    assert module.analyze_faces(analysis, np.zeros((2, 2, 3), dtype=np.uint8)) == [face]
    assert analysis.sizes == [(640, 640), (576, 576), (512, 512), (448, 448), (384, 384), (320, 320)]
