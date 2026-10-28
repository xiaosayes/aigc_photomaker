# Server-pending: run these dependency/behavior tests on the GPU server.
from aigc_photomaker.errors import AppError


def test_error_payload_does_not_include_stack():
    error = AppError(400, "INVALID_IMAGE", "图片无法解码", False)
    assert error.to_dict() == {
        "code": "INVALID_IMAGE",
        "message": "图片无法解码",
        "retryable": False,
    }
