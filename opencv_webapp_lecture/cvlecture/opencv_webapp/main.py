#############################
# Python Image Pixelator    #
# openCV library            #
#############################

from PIL import Image
import numpy as np
import cv2

def fill(op, pos, n, avg):
    # Fills the new image pixels with the average values
    y, x = pos
    for i in range(x, x + n):
        for j in range(y, y + n):
            op.putpixel((i, j), avg)


# Calculates average RGB values of the n*n pixel matrix
def pixel_avg(square, n):
    r, g, b = 0, 0, 0
    for x in range(n):
        for y in range(n):
            r += square[x][y][2]
            g += square[x][y][1]
            b += square[x][y][0]
    r, g, b = r // (n * n), g // (n * n), b // (n * n)
    return (r, g, b)


def pixelize(img, path):

    #img = cv2.imread(path, 1)  # img 역할

    if (type(img) is np.ndarray):



        im_height, im_width = img.shape[:2]

        # Tweak this parameter
        ######
        n = 15  # This is the pixelation factor
        ######

        # These pixels are ignored for now
        rem_width = im_width % n
        rem_height = im_height % n
        print(im_height, im_width)

        # The output image
        # Red pixels in the end result denote the unprocessed pixels
        op = Image.new('RGB', (im_width, im_height))

        for x in range(0, im_height - rem_height, n):
            for y in range(0, im_width - rem_width, n):
                avg = pixel_avg(img[x:x + n, y:y + n], n)
                fill(op, (x, y), n, avg)

        op.save(path)
        print(op)


    else:
        print('someting error')
        print(path)

def pixelizes(path):

    img = cv2.imread(path, 1)

    if (type(img) is np.ndarray):



        im_height, im_width = img.shape[:2]

        # Tweak this parameter
        ######
        n = 15  # This is the pixelation factor
        ######

        # These pixels are ignored for now
        rem_width = im_width % n
        rem_height = im_height % n
        print(im_height, im_width)

        # The output image
        # Red pixels in the end result denote the unprocessed pixels
        op = Image.new('RGB', (im_width, im_height))

        for x in range(0, im_height - rem_height, n):
            for y in range(0, im_width - rem_width, n):
                avg = pixel_avg(img[x:x + n, y:y + n], n)
                fill(op, (x, y), n, avg)

        op.save(path)
        print(op)


    else:
        print('someting error')
        print(path)


if __name__ == '__main__':
    pixelize()
