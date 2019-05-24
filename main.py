#############################
# Python Image Pixelator    #
# openCV library            #
#############################

from PIL import Image
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


def main():
    name = 'yesol.jpg'  # give your filename here
    pixels = cv2.imread(name)  # img 역할

    im_height, im_width = pixels.shape[:2]

    # Tweak this parameter
    ######
    n = 15  # This is the pixelation factor
    ######

    # These pixels are ignored for now
    rem_width = im_width % n
    rem_height = im_height % n
    print(im_height, im_width)

    # The ouput image
    # Red pixels in the end result denote the unprocessed pixels
    op = Image.new('RGB', (im_width, im_height), color='red')

    for x in range(0, im_height - rem_height, n):
        for y in range(0, im_width - rem_width, n):
            avg = pixel_avg(pixels[x:x + n, y:y + n], n)
            fill(op, (x, y), n, avg)

    op.show()
    op.save(name.split('.')[0] + '_pixel.jpg')


if __name__ == '__main__':
    main()
