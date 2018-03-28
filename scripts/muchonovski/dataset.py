import os
import random
from PIL import Image

IMAGE_SIZE = 48


class Generator:
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.dataDir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        self.bans = ['dirty', 'gohan', 'kaya_a', 'kaya_b', 'kaya_c', 'kaya_d', 'muji', 'oritatami', 'paper', 'stripe']
        self.masus = ['dot_xy', 'dot', 'handwriting', 'nodot_xy', 'nodot']
        self.komas = ['dirty', 'kinki', 'kinki_torafu', 'ryoko', 'ryoko_torafu']
        self.pieceMap = {
            'Gfu':   'W_FU', 'Sfu':   'B_FU',
            'Gto':   'W_TO', 'Sto':   'B_TO',
            'Gkyo':  'W_KY', 'Skyo':  'B_KY',
            'Gnkyo': 'W_NY', 'Snkyo': 'B_NY',
            'Gkei':  'W_KE', 'Skei':  'B_KE',
            'Gnkei': 'W_NK', 'Snkei': 'B_NK',
            'Ggin':  'W_GI', 'Sgin':  'B_GI',
            'Gngin': 'W_NG', 'Sngin': 'B_NG',
            'Gkin':  'W_KI', 'Skin':  'B_KI',
            'Gkaku': 'W_KA', 'Skaku': 'B_KA',
            'Guma':  'W_UM', 'Suma':  'B_UM',
            'Ghi':   'W_HI', 'Shi':   'B_HI',
            'Gryu':  'W_RY', 'Sryu':  'B_RY',
            'Gou':   'W_OU', 'Sou':   'B_OU',
            None: 'BLANK',
        }

    def run(self):
        for piece in self.pieceMap.keys():
            saveDir = os.path.join(self.dataDir, self.pieceMap[piece])
            for i in range(10):
                ban = random.choice(self.bans)
                masu = random.choice(self.masus)
                koma = random.choice(self.komas)
                img = self.generate(ban, masu, koma, piece)
                filename = 'muchonovski_{:02d}.jpg'.format(i)
                savePath = os.path.join(saveDir, filename)
                print('{}: {}...'.format(os.path.basename(saveDir), filename))
                with open(savePath, 'w') as fp:
                    img.convert('RGB').save(fp, quality=90)

    def generate(self, banName, masuName, komaName, piece):
        file, rank = random.randrange(9), random.randrange(9)
        offset = (11 + 43 * file, 11 + 48 * rank)

        ban = Image.open(os.path.join(self.imageDir, 'ban', 'ban_{}.png'.format(banName)))
        masu = Image.open(os.path.join(self.imageDir, 'masu', 'masu_{}.png'.format(masuName)))
        ban.alpha_composite(masu)
        if piece is not None:
            koma = Image.open(os.path.join(self.imageDir, 'koma', 'koma_{}'.format(komaName), '{}.png'.format(piece)))
            ban.alpha_composite(koma, dest=offset)
        return ban.crop(box=(
            offset[0] - int((IMAGE_SIZE - 43) / 2),
            offset[1] - int((IMAGE_SIZE - 48) / 2),
            offset[0] - int((IMAGE_SIZE - 43) / 2) + IMAGE_SIZE,
            offset[1] - int((IMAGE_SIZE - 48) / 2) + IMAGE_SIZE))


if __name__ == '__main__':
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'muchonovski')
    Generator(imageDir).run()
