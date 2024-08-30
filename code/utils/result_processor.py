from lib.farancia import IImage


def convert_range(video, output_range, input_range=None):
    if input_range is None:
        if video.min() < 0:
            input_range = [-1, 1]
        elif video.max() > 1:
            input_range = [0, 255]
        else:
            input_range = [0, 1]
    video = (video-input_range[0])/(input_range[1]-input_range[0])  # [0,1]
    video = video * (output_range[1]-output_range[0]) + output_range[0]
    return video


def concat_chunks(result_chunks):
    if not isinstance(result_chunks, list):
        result_chunks = [result_chunks]
    concatenated_result = None

    for chunk in result_chunks:
        assert chunk.min() >= 0
        chunk = IImage(chunk, vmin=0, vmax=255)
        if concatenated_result is None:
            concatenated_result = chunk
        else:
            concatenated_result &= chunk
    return concatenated_result
