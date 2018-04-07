import os
import random
from PIL import Image

IMAGE_SIZE = 48


class Generator:
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        self.pieceMap = {
            'syougi01_ousyou':    'OU',
            'syougi02_gyokusyou': 'OU',
            'syougi03_hisya':     'HI',
            'syougi04_ryuuou':    'RY',
            'syougi05_gakugyou':  'KA',
            'syougi06_ryuuma':    'UM',
            'syougi07_kinsyou':   'KI',
            'syougi08_ginsyou':   'GI',
            'syougi09_narigin':   'NG',
            'syougi10_keima':     'KE',
            'syougi11_narikei':   'NK',
            'syougi12_kyousya':   'KY',
            'syougi13_narikyou':  'NY',
            'syougi14_fuhyou':    'FU',
            'syougi15_tokin':     'TO',
            None:                 'BLANK',
        }

    def run(self):
        for piece in self.pieceMap.keys():
            for opposite in [True, False]:
                prefix = 'W_' if opposite else 'B_'
                if piece is None:
                    prefix = ''
                saveDir = os.path.join(self.dataDir, prefix + self.pieceMap[piece])
                for i in range(3):
                    img = self.generate(piece, opposite)
                    filename = 'irasutoya_{:02d}.jpg'.format(i)
                    savePath = os.path.join(saveDir, filename)
                    print('{}: {}...'.format(os.path.basename(saveDir), filename))
                    with open(savePath, 'w') as fp:
                        img.convert('RGB').save(fp, quality=random.randint(90, 100))
                if piece is None:
                    break

    def generate(self, piece, opposite):
        file, rank = random.randrange(9), random.randrange(9)
        xStep, yStep = 430.0 / 9.0, 470.0 / 9.0
        offset = (14 + xStep * file, 14 + yStep * rank)

        ban = Image.open(os.path.join(self.imageDir, 'syougi_ban.png'))
        if piece:
            koma = Image.open(os.path.join(self.imageDir, '{}.png'.format(piece))).resize((41, 50))
            if opposite:
                koma = koma.rotate(180)
            komaOffset = (int(offset[0] + 4.5), int(offset[1] + 2))
            ban.alpha_composite(koma, dest=komaOffset)
        return ban.crop(box=(
            offset[0] + (xStep - yStep) / 2,
            offset[1],
            offset[0] + (xStep + yStep) / 2,
            offset[1] + yStep)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)


if __name__ == '__main__':
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'irasutoya')
    Generator(imageDir).run()
