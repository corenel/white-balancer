# -*- coding: utf-8 -*-

from PIL import Image
import colorcorrect.algorithm as cca
import numpy as np
import sys


def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.array(pimg)[:]
    # nimg.flags.writeable = True
    return nimg


def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))


if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    # img.show()
    to_pil(cca.stretch(from_pil(img)))
    to_pil(cca.grey_world(from_pil(img)))
    to_pil(cca.retinex(from_pil(img)))
    to_pil(cca.max_white(from_pil(img)))
    to_pil(cca.retinex_with_adjust(cca.retinex(from_pil(img))))
    to_pil(cca.standard_deviation_weighted_grey_world(from_pil(img), 20, 20))
    to_pil(
        cca.standard_deviation_and_luminance_weighted_gray_world(
            from_pil(img), 20, 20))
    to_pil(cca.luminance_weighted_gray_world(from_pil(img), 20, 20))
    to_pil(cca.automatic_color_equalization(from_pil(img)))
