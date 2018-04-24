#!/usr/bin/env python
import os
import random
from PIL import Image

IMAGE_SIZE = 96


class Generator:
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        self.pieceMap = {
            'B_FU': ['sgl08'], 'B_TO': ['sgl18', 'sgl28'],
            'B_KY': ['sgl07'], 'B_NY': ['sgl17', 'sgl27'],
            'B_KE': ['sgl06'], 'B_NK': ['sgl16', 'sgl26'],
            'B_GI': ['sgl05'], 'B_NG': ['sgl15', 'sgl25'],
            'B_KI': ['sgl04'],
            'B_KA': ['sgl03'], 'B_UM': ['sgl13', 'sgl23'],
            'B_HI': ['sgl02'], 'B_RY': ['sgl12', 'sgl22'],
            'B_OU': ['sgl01', 'sgl11'],
            'W_FU': ['sgl38'], 'W_TO': ['sgl48', 'sgl58'],
            'W_KY': ['sgl37'], 'W_NY': ['sgl47', 'sgl57'],
            'W_KE': ['sgl36'], 'W_NK': ['sgl46', 'sgl56'],
            'W_GI': ['sgl35'], 'W_NG': ['sgl45', 'sgl55'],
            'W_KI': ['sgl34'],
            'W_KA': ['sgl33'], 'W_UM': ['sgl43', 'sgl53'],
            'W_HI': ['sgl32'], 'W_RY': ['sgl42', 'sgl51'],
            'W_OU': ['sgl31', 'sgl41'],
            'BLANK': [None]
        }

    def run(self):
        otherIndex = 0
        for k, v in self.pieceMap.items():
            for i, piece in enumerate(v):
                for j in range(3):
                    img, other = self.generate(piece)
                    fileName = 'sozai_{:02d}.jpg'.format(i * 3 + j)
                    savePath = os.path.join(self.dataDir, k, fileName)
                    print('{}: {}...'.format(k, fileName))
                    with open(savePath, 'w') as fp:
                        img.convert('RGB').save(fp, quality=random.randrange(80, 100))
                    if random.randrange(21) == 0:
                        fileName = 'sozai_{:02d}.jpg'.format(otherIndex)
                        otherPath = os.path.join(self.dataDir, 'OTHER', fileName)
                        print('OTHER: {}...'.format(fileName))
                        with open(otherPath, 'w') as fp:
                            other.convert('RGB').save(fp, quality=random.randint(80, 100))
                        otherIndex += 1

    def generate(self, piece):
        boardPath = os.path.join(self.imageDir, 'board', 'japanese-chess-b02.jpg')
        board = Image.open(boardPath)
        img = Image.new('RGBA', board.size, color='blue')
        img.paste(board, box=(0, 0, board.width, board.height))

        file, rank = random.randrange(9), random.randrange(9)
        if piece is not None:
            piecePath = os.path.join(self.imageDir, 'koma', '60x64', '{}.png'.format(piece))
            piece = Image.open(piecePath)
            img.alpha_composite(piece, dest=(file * 60 + 30, rank * 64 + 30))
        otherOffset = [file * 60 + 30 - 2, rank * 64 + 30]
        if random.randrange(2) == 0:
            otherOffset[0] += random.choice([30, -30])
            otherOffset[1] = random.randrange(img.height - 64)
        else:
            otherOffset[0] = random.randrange(img.width - 64)
            otherOffset[1] += random.choice([32, -32])
        resample = random.choice([Image.NEAREST, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS])
        return [
            img.crop(box=(
                file * 60 + 30 - 2,
                rank * 64 + 30,
                file * 60 + 30 + 62,
                rank * 64 + 30 + 64)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=resample),
            img.crop(box=(
                otherOffset[0],
                otherOffset[1],
                otherOffset[0] + 64,
                otherOffset[1] + 64)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=resample),
        ]


if __name__ == '__main__':
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sozai', 'japanese-chess')
    Generator(imageDir).run()
