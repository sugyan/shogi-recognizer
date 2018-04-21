#!/usr/bin/env python
import glob
import os
import random
from PIL import Image, ImageDraw

IMAGE_SIZE = 48


class Generator:
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        self.backgroundColor = (255, 209, 122)
        self.pieceMap = {
            'B_FU': (7, 0), 'W_FU': (7, 3),
            'B_TO': (7, 1), 'W_TO': (7, 4),
            'B_KY': (6, 0), 'W_KY': (6, 3),
            'B_NY': (6, 1), 'W_NY': (6, 4),
            'B_KE': (5, 0), 'W_KE': (5, 3),
            'B_NK': (5, 1), 'W_NK': (5, 4),
            'B_GI': (4, 0), 'W_GI': (4, 3),
            'B_NG': (4, 1), 'W_NG': (4, 4),
            'B_KI': (3, 0), 'W_KI': (3, 3),
            'B_KA': (2, 0), 'W_KA': (2, 3),
            'B_UM': (2, 1), 'W_UM': (2, 4),
            'B_HI': (1, 0), 'W_HI': (1, 3),
            'B_RY': (1, 1), 'W_RY': (1, 4),
            'B_OU': (0, 0), 'W_OU': [0, 3],
            'BLANK': None,
        }

    def run(self):
        otherIndex = 0
        for i, bmp in enumerate(glob.glob(os.path.join(self.imageDir, '*.bmp'))):
            for k, v in self.pieceMap.items():
                if k == 'W_OU' and random.randrange(2) == 0:
                    v[1] += 1
                for j in range(2):
                    fileName = 'shineleckoma_{:03d}.jpg'.format(i * 2 + j)
                    savePath = os.path.join(self.dataDir, k, fileName)
                    print('{}: {}...'.format(os.path.basename(k), fileName))
                    img, other = self.generate(bmp, v)
                    with open(savePath, 'w') as fp:
                        img.save(fp, quality=random.randint(90, 100))
                    if random.randrange(28 * 2) == 0:
                        fileName = 'shineleckoma_{:03d}.jpg'.format(otherIndex)
                        otherPath = os.path.join(self.dataDir, 'OTHER', fileName)
                        print('OTHER: {}...'.format(otherPath))
                        with open(otherPath, 'w') as fp:
                            other.save(fp, quality=random.randint(90, 100))
                        otherIndex += 1

    def generate(self, bmp, loc):
        img = Image.open(bmp)
        board = Image.new('RGB', (463, 472), color=self.backgroundColor)
        draw = ImageDraw.Draw(board)
        for i in range(10):
            offset, width = 0, 1
            if i == 0 or i == 9:
                width = 2
                if i == 0:
                    offset = -1
            draw.line(((6 + 50 * i + offset, 5), (6 + 50 * i + offset, 466)), fill=0, width=width)
            draw.line(((5, 6 + 51 * i + offset), (457, 6 + 51 * i + offset)), fill=0, width=width)
        for pt in ((156, 159), (156, 312), (306, 159), (306, 312)):
            draw.ellipse(((pt[0] - 2, pt[1] - 2), (pt[0] + 2, pt[1] + 2)), fill=0)
        file, rank = random.randrange(9), random.randrange(9)
        if loc is not None:
            piece = img.crop(box=(loc[0] * 43, loc[1] * 48, (loc[0] + 1) * 43, (loc[1] + 1) * 48))
            board.paste(piece, box=(
                file * 50 + 10, rank * 51 + 8,
                file * 50 + 53, rank * 51 + 56))
        otherOffset = [file * 50 + 5, rank * 51 + 5]
        if random.randrange(2) == 0:
            otherOffset[0] += random.choice([25, -25])
            otherOffset[1] = random.randrange(472 - 51)
        else:
            otherOffset[0] = random.randrange(463 - 50)
            otherOffset[1] += random.choice([25.5, -25.5])
        return [
            board.crop(box=(
                file * 50 + 5,
                rank * 51 + 5,
                file * 50 + 58,
                rank * 51 + 59)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR),
            board.crop(box=(
                otherOffset[0],
                otherOffset[1],
                otherOffset[0] + 53,
                otherOffset[1] + 54)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR),
        ]


if __name__ == '__main__':
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'shineleckoma')
    Generator(imageDir).run()
