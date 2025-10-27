### execute frontal face filter

from facefilter.loader import ImageLoader

loader = ImageLoader("/Users/xiangxzou/Desktop/nn_FACE")

for path, img, meta in loader.iter_images():
    print(meta.width, meta.height, meta.ext)
