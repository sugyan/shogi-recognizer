import argparse
import os
import random
import re
from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 48


class Generator:
    def __init__(self, fonts):
        self.dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        self.fonts = fonts
        self.pieceMap = {
            'FU': ['歩'],
            'TO': ['と'],
            'KY': ['香'],
            'NY': ['杏', '成香'],
            'KE': ['桂'],
            'NK': ['圭', '成桂'],
            'GI': ['銀'],
            'NG': ['全', '成銀'],
            'KI': ['金'],
            'KA': ['角'],
            'UM': ['馬'],
            'HI': ['飛'],
            'RY': ['龍', '竜'],
            'OU': ['玉', '王'],
            'BLANK': [None],
        }

    def run(self):
        otherIndex = 0
        for piece in self.pieceMap.keys():
            for opposite in [True, False]:
                prefix = 'W_' if opposite else 'B_'
                if piece == 'BLANK':
                    prefix = ''
                saveDir = os.path.join(self.dataDir, prefix + piece)
                i = 0
                for font in self.fonts:
                    for char in self.pieceMap[piece]:
                        for _ in range(5):
                            fileName = 'character_{:02d}.jpg'.format(i)
                            savePath = os.path.join(saveDir, fileName)
                            print('{}: {}...'.format(os.path.basename(saveDir), fileName))
                            img, other = self.generate(char, font, opposite)
                            with open(savePath, 'w') as fp:
                                img.save(fp, quality=random.randint(90, 100))
                            i += 1
                            if random.randrange(20) == 0:
                                fileName = 'character_{:02d}.jpg'.format(otherIndex)
                                otherPath = os.path.join(self.dataDir, 'OTHER', fileName)
                                print('OTHER: {}...'.format(otherPath))
                                with open(otherPath, 'w') as fp:
                                    other.save(fp, quality=random.randint(90, 100))
                                otherIndex += 1

    def generate(self, char, font, opposite):
        size = random.randrange(600, 900)
        step = size / 10.0
        img = Image.new('RGB', (size, size), color='white')
        draw = ImageDraw.Draw(img)
        for i in range(10):
            width = 1
            if i == 0 or i == 9:
                width = 2
            draw.line([(step * (i + 0.5), step * 0.5), (step * (i + 0.5), step * 9.5)], fill=0, width=width)
            draw.line([(step * 0.5, step * (i + 0.5)), (step * 9.5, step * (i + 0.5))], fill=0, width=width)
        pieceSize = step * random.randrange(80, 95) / 100.0
        pieceImg = Image.new('RGB', (int(pieceSize), int(pieceSize)), color='white')
        if char is not None:
            font = ImageFont.truetype(font, size=int(pieceSize))
            pieceDraw = ImageDraw.Draw(pieceImg)
            if char.startswith(u'成'):
                pieceDraw.text((0, 0), char[1], fill=0, font=font)
                nariImg = Image.new('RGB', (int(pieceSize), int(pieceSize) * 2), color='white')
                nariDraw = ImageDraw.Draw(nariImg)
                nariDraw.text((0, 0), u'成', fill=0, font=font)
                nariImg.paste(pieceImg, box=(0, int(pieceSize)))
                pieceImg = nariImg.resize((pieceImg.width, pieceImg.height))
            else:
                pieceDraw.text((0, 0), char, fill=0, font=font)
            if opposite:
                pieceImg = pieceImg.rotate(180)
        file, rank = random.randrange(9), random.randrange(9)
        img.paste(pieceImg, box=(
            int(step * (file + 1) - pieceSize * 0.5) + 1,
            int(step * (rank + 1) - pieceSize * 0.5) + 1))
        otherOffset = [step * (file + 0.5) - 2, step * (rank + 0.5) - 2]
        if random.randrange(2) == 0:
            otherOffset[0] += step * (random.randrange(2) - 0.5)
            otherOffset[1] = random.randrange(size - int(step))
        else:
            otherOffset[0] = random.randrange(size - int(step))
            otherOffset[1] += step * (random.randrange(2) - 0.5)
        return [
            img.crop(box=(
                step * (file + 0.5) - 2,
                step * (rank + 0.5) - 2,
                step * (file + 1.5) + 2,
                step * (rank + 1.5) + 2)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR),
            img.crop(box=(
                otherOffset[0],
                otherOffset[1],
                otherOffset[0] + step + 2,
                otherOffset[1] + step + 2)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR),
        ]


if __name__ == '__main__':
    fontFilter = re.compile('W[6-9]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--fonts', nargs='*', required=True)
    args = parser.parse_args()
    Generator([f for f in args.fonts if not fontFilter.search(f)]).run()
