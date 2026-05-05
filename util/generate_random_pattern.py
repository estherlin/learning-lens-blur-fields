

import cv2
import numpy as np


##########################################################################
def generate_pt(pimgh, pimgw, fre, num=300, enlarge=1.2):
    """
    Generate random patterns of different frequencies
    :param pimgh: height of the pattern
    :param pimgw: width of the pattern
    :param fre: frequency of the pattern
    :param num: number of patterns
    :param enlarge: enlarge the pattern
    :return: list of patterns
    """

    pimghbig, pimgwbig = int(pimgh * enlarge), int(pimgw * enlarge)
    # pimghbig, pimgwbig = pimgh, pimgw
    ims_hxw = np.random.rand(pimghbig, pimgwbig).astype(np.float32)

    imfre = np.fft.fft2(ims_hxw)
    imfreshift = np.fft.fftshift(imfre)
    centy, centx = pimghbig // 2, pimgwbig // 2
    dc = imfreshift[centy, centx]
    # fre = 32
    imfreshift[centy - fre:centy + fre + 1, centx - fre:centx + fre + 1] = 0
    imfreshift[:centy - 2 * fre] = 0
    imfreshift[centy + 2 * fre + 1:] = 0
    imfreshift[:, :centx - 2 * fre] = 0
    imfreshift[:, centx + 2 * fre + 1:] = 0
    imfresamp = np.sqrt(imfreshift * np.conj(imfreshift))
    imfreshift = imfreshift / (1e-10 + imfresamp)

    imfreshift[centy, centx] = dc
    imfre = np.fft.fftshift(imfreshift)
    im = np.real(np.fft.ifft2(imfre))

    immean = im.mean()
    im[im < immean] = 0
    im[im >= immean] = 1

    start_y, start_x = (pimghbig - pimgh) // 2, (pimgwbig - pimgw) // 2
    im = im[start_y:start_y + pimgh, start_x:start_x + pimgw]
    ims = []
    for d in range(num):
        shiftx = np.random.randint(pimgw)
        shifty = np.random.randint(pimgh)
        imshift = np.roll(im, shift=shiftx, axis=0)
        imshift = np.roll(imshift, shift=shifty, axis=1)
        ims.append(imshift)

    # ims = [np.roll(im, shift=d, axis=0) for d in range(150)]
    '''
    for i in range(300):
        cv2.imshow('', (ims[i] * 255).astype(np.uint8))
        cv2.waitKey(0)
    '''

    return ims


if __name__ == '__main__':

    # Generate random patterns of different frequencies
    for freq in [5, 7, 10, 20, 50, 70, 100]:
        # Testing set
        ims = generate_pt(pimgh=1512, pimgw=2016, fre=freq, num=1, enlarge=1.0)
        for im in ims:
            cv2.imwrite('./pixel/train_patterns/freq-{}.png'.format(freq), (im * 255).astype(np.uint8))

        # Training set
        ims = generate_pt(pimgh=1512, pimgw=2016, fre=freq, num=1, enlarge=1.0)
        for im in ims:
            cv2.imwrite('./pixel/test_patterns/freq-{}.png'.format(freq), (im * 255).astype(np.uint8))
