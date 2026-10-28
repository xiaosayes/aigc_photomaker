def select_single_face(faces):
    if not faces:
        raise ValueError("NO_FACE_DETECTED: upload one clear, visible face")
    if len(faces) != 1:
        raise ValueError("MULTIPLE_FACES_DETECTED: upload an image containing exactly one face")
    return faces[0]


def select_single_output_face(faces):
    if not faces:
        raise ValueError("OUTPUT_NO_FACE_DETECTED: generated image has no detectable face")
    if len(faces) != 1:
        raise ValueError(
            "OUTPUT_MULTIPLE_FACES_DETECTED: generated image must contain exactly one face"
        )
    return faces[0]
