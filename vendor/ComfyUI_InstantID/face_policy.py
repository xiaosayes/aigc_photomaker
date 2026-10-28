"""Single-face input/output guards for the InstantID ComfyUI workflow."""


def validate_face_count(faces, output=False):
    prefix = "OUTPUT_" if output else ""
    if not faces:
        raise ValueError(f"{prefix}NO_FACE_DETECTED: expected exactly one face")
    if len(faces) != 1:
        raise ValueError(f"{prefix}MULTIPLE_FACES_DETECTED: expected exactly one face")
    return faces[0]


def analyze_faces(faceanalysis, image):
    """Return all faces from one BGR image, retrying smaller detection inputs."""
    for size in range(640, 256, -64):
        faceanalysis.det_model.input_size = (size, size)
        faces = faceanalysis.get(image)
        if faces:
            return faces
    faceanalysis.det_model.input_size = (256, 256)
    return faceanalysis.get(image) or []


class _SingleFaceGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "faceanalysis": ("FACEANALYSIS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "InstantID"

    def _validate(self, images, faceanalysis, output):
        from .utils import tensor_to_image

        for image in tensor_to_image(images):
            validate_face_count(analyze_faces(faceanalysis, image), output=output)
        return (images,)


class InstantIDSingleFaceInputGuard(_SingleFaceGuard):
    FUNCTION = "validate_input"

    def validate_input(self, images, faceanalysis):
        return self._validate(images, faceanalysis, output=False)


class InstantIDSingleFaceOutputGuard(_SingleFaceGuard):
    FUNCTION = "validate_output"

    def validate_output(self, images, faceanalysis):
        return self._validate(images, faceanalysis, output=True)


NODE_CLASS_MAPPINGS = {
    "InstantIDSingleFaceInputGuard": InstantIDSingleFaceInputGuard,
    "InstantIDSingleFaceOutputGuard": InstantIDSingleFaceOutputGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDSingleFaceInputGuard": "InstantID Single Face Input Guard",
    "InstantIDSingleFaceOutputGuard": "InstantID Single Face Output Guard",
}
