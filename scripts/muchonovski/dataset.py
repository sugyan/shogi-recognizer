import os
from PIL import Image


class Generator:
    def __init__(self, imageDir, dataDir):
        self.imageDir = imageDir
        self.dataDir = dataDir

    def run(self):
        img = Image.open(os.path.join(self.imageDir, 'koma', 'koma_dirty', 'Gfu.png'))
        savePath = os.path.join(self.dataDir, 'W_FU', '1.jpg')
        with open(savePath, 'w') as fp:
            img.convert('RGB').save(fp)


if __name__ == '__main__':
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'muchonovski')
    dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    Generator(imageDir, dataDir).run()
