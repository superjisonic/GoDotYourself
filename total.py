from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PIL import Image
import cv2

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

def fill(op, pos, n, avg):
    # Fills the new image pixels with the average values
    y, x = pos
    for i in range(x, x + n):
        for j in range(y, y + n):
            op.putpixel((i, j), avg)

def main():
    name = 'yesol.jpg'  # give your filename here
    pixels = cv2.imread(name)  # img 역할

    im_height, im_width = pixels.shape[:2]

    # Tweak this parameter
    n = 30  # This is the pixelation factor

    # These pixels are ignored for now
    rem_width = im_width % n
    rem_height = im_height % n
    print(im_height, im_width)

    # The ouput image
    # Red pixels in the end result denote the unprocessed pixels
    output = Image.new('RGB', (im_width, im_height), color='red')

    for x in range(0, im_height - rem_height, n):
        for y in range(0, im_width - rem_width, n):
            avg = pixel_avg(pixels[x:x + n, y:y + n], n)
            fill(output, (x, y), n, avg)
    output.save(name.split('.')[0] + '_pixel.jpg')

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('gui.ui', self)


        # 메인이미지
        upload_img = self.label  # ui에 저장된 오브젝트
        pixmap = QPixmap("yesol.jpg")
        upload_img.setPixmap(pixmap)
        upload_img.setFixedSize(upload_img.width(), upload_img.height())
        upload_img.setScaledContents(True)

        # ui에 있는 버튼 클릭시-어떻게 참조 하는거지?
        self.accept.clicked.connect(self.clickMethod1)
        self.reset.clicked.connect(self.clickMethod2)


    def clickMethod1(self):
        convert_img = self.label_2
        pixmap = QPixmap("yesol_pixel.jpg")
        convert_img.setPixmap(pixmap)
        convert_img.setFixedSize(convert_img.width(), convert_img.height())
        convert_img.setScaledContents(True)

    def clickMethod2(self):
        sys.exit(app.exec_())


if __name__ == '__main__':
    import sys
    main()
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())