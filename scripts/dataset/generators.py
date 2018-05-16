import os
import random
from PIL import Image, ImageDraw, ImageFont
from generator import Generator

IMAGE_SIZE = 96
RESAMPLES = (Image.NEAREST, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS)


def add_noise(img):
    sigma = random.uniform(0, 5)
    for i in range(img.width * img.height):
        x, y = int(i / img.height), i % img.height
        p = [int(v + random.normalvariate(0, sigma)) for v in img.getpixel((x, y))]
        img.putpixel((x, y), tuple(p))


class AlphaGenerator(Generator):
    def __init__(self, imageDir):
        self.imageDir = imageDir
        self.pieceMap = {
            'irasutoya': {
                'B_FU': ['syougi14_fuhyou'],   'W_FU': ['syougi14_fuhyou'],
                'B_TO': ['syougi15_tokin'],    'W_TO': ['syougi15_tokin'],
                'B_KY': ['syougi12_kyousya'],  'W_KY': ['syougi12_kyousya'],
                'B_NY': ['syougi13_narikyou'], 'W_NY': ['syougi13_narikyou'],
                'B_KE': ['syougi10_keima'],    'W_KE': ['syougi10_keima'],
                'B_NK': ['syougi11_narikei'],  'W_NK': ['syougi11_narikei'],
                'B_GI': ['syougi08_ginsyou'],  'W_GI': ['syougi08_ginsyou'],
                'B_NG': ['syougi09_narigin'],  'W_NG': ['syougi09_narigin'],
                'B_KI': ['syougi07_kinsyou'],  'W_KI': ['syougi07_kinsyou'],
                'B_KA': ['syougi05_gakugyou'], 'W_KA': ['syougi05_gakugyou'],
                'B_UM': ['syougi06_ryuuma'],   'W_UM': ['syougi06_ryuuma'],
                'B_HI': ['syougi03_hisya'],    'W_HI': ['syougi03_hisya'],
                'B_RY': ['syougi04_ryuuou'],   'W_RY': ['syougi04_ryuuou'],
                'B_OU': ['syougi01_ousyou', 'syougi02_gyokusyou'],
                'W_OU': ['syougi01_ousyou', 'syougi02_gyokusyou'],
            },
            'muchonovski': {
                'B_FU': ['Sfu'],   'W_FU': ['Gfu'],
                'B_TO': ['Sto'],   'W_TO': ['Gto'],
                'B_KY': ['Skyo'],  'W_KY': ['Gkyo'],
                'B_NY': ['Snkyo'], 'W_NY': ['Gnkyo'],
                'B_KE': ['Skei'],  'W_KE': ['Gkei'],
                'B_NK': ['Snkei'], 'W_NK': ['Gnkei'],
                'B_GI': ['Sgin'],  'W_GI': ['Ggin'],
                'B_NG': ['Sngin'], 'W_NG': ['Gngin'],
                'B_KI': ['Skin'],  'W_KI': ['Gkin'],
                'B_KA': ['Skaku'], 'W_KA': ['Gkaku'],
                'B_UM': ['Suma'],  'W_UM': ['Guma'],
                'B_HI': ['Shi'],   'W_HI': ['Ghi'],
                'B_RY': ['Sryu'],  'W_RY': ['Gryu'],
                'B_OU': ['Sou'],   'W_OU': ['Gou'],
            },
            'sozai': {
                'B_FU': ['sgl08'],          'W_FU': ['sgl38'],
                'B_TO': ['sgl18', 'sgl28'], 'W_TO': ['sgl48', 'sgl58'],
                'B_KY': ['sgl07'],          'W_KY': ['sgl37'],
                'B_NY': ['sgl17', 'sgl27'], 'W_NY': ['sgl47', 'sgl57'],
                'B_KE': ['sgl06'],          'W_KE': ['sgl36'],
                'B_NK': ['sgl16', 'sgl26'], 'W_NK': ['sgl46', 'sgl56'],
                'B_GI': ['sgl05'],          'W_GI': ['sgl35'],
                'B_NG': ['sgl15', 'sgl25'], 'W_NG': ['sgl45', 'sgl55'],
                'B_KI': ['sgl04'],          'W_KI': ['sgl34'],
                'B_KA': ['sgl03'],          'W_KA': ['sgl33'],
                'B_UM': ['sgl13', 'sgl23'], 'W_UM': ['sgl43', 'sgl53'],
                'B_HI': ['sgl02'],          'W_HI': ['sgl32'],
                'B_RY': ['sgl12', 'sgl22'], 'W_RY': ['sgl42', 'sgl51'],
                'B_OU': ['sgl01', 'sgl11'], 'W_OU': ['sgl31', 'sgl41'],
            }
        }

    def generate(self, piece):
        board, xywh = self.__board()
        # draw piece
        file, rank = random.randrange(9), random.randrange(9)
        if piece != 'BLANK':
            piece = self.__piece(piece)
            resized = piece.resize([int(x) for x in xywh[2:4]], resample=random.choice(RESAMPLES))
            board.alpha_composite(resized, dest=(int(xywh[0] + file * xywh[2]), int(xywh[1] + rank * xywh[3])))
        # crop
        box = (
            xywh[0] + (file + random.normalvariate(0.0, 0.02)) * xywh[2],
            xywh[1] + (rank + random.normalvariate(0.0, 0.02)) * xywh[3],
            xywh[0] + (file + random.normalvariate(0.0, 0.02) + 1) * xywh[2],
            xywh[1] + (rank + random.normalvariate(0.0, 0.02) + 1) * xywh[3],
        )
        cropped = board.crop(box=box)
        # resize
        add_noise(cropped)
        return cropped.resize((IMAGE_SIZE, IMAGE_SIZE), resample=random.choice(RESAMPLES)).convert('RGB')

    def __board(self):
        source = random.choices(['irasutoya', 'muchonovski', 'sozai'], weights=[1, 10, 1])[0]
        path = random.choice({
            'irasutoya': [os.path.join('irasutoya', 'syougi_ban.png')],
            'muchonovski': [
                os.path.join('muchonovski', 'ban', 'ban_dirty.png'),
                os.path.join('muchonovski', 'ban', 'ban_gohan.png'),
                os.path.join('muchonovski', 'ban', 'ban_kaya_a.png'),
                os.path.join('muchonovski', 'ban', 'ban_kaya_b.png'),
                os.path.join('muchonovski', 'ban', 'ban_kaya_c.png'),
                os.path.join('muchonovski', 'ban', 'ban_kaya_d.png'),
                os.path.join('muchonovski', 'ban', 'ban_muji.png'),
                os.path.join('muchonovski', 'ban', 'ban_oritatami.png'),
                os.path.join('muchonovski', 'ban', 'ban_paper.png'),
                os.path.join('muchonovski', 'ban', 'ban_stripe.png'),
            ],
            'sozai': [
                os.path.join('sozai', 'japanese-chess', 'board', 'japanese-chess-b02.jpg'),
            ]
        }[source])
        board = Image.open(os.path.join(self.imageDir, path))
        if board.mode != 'RGBA':
            img = Image.new('RGBA', board.size, color='white')
            img.paste(board, box=(0, 0, board.width, board.height))
            board = img
        if source == 'muchonovski':
            paths = [
                os.path.join('muchonovski', 'masu', 'masu_dot.png'),
                os.path.join('muchonovski', 'masu', 'masu_handwriting.png'),
                os.path.join('muchonovski', 'masu', 'masu_nodot.png'),
            ]
            grid = Image.open(os.path.join(self.imageDir, random.choice(paths)))
            board.alpha_composite(grid)
        return board, {
            'irasutoya': [16.0, 15.0, 430.0 / 9.0, 470.0 / 9.0],
            'muchonovski': [11.0, 11.0, 43.0, 48.0],
            'sozai': [30.0, 30.0, 60.0, 64.0]
        }[source]

    def __piece(self, piece):
        source = random.choices(['irasutoya', 'muchonovski', 'sozai'], weights=[1, 5, 1])[0]
        path = os.path.join(self.imageDir, source)
        if source == 'muchonovski':
            path = os.path.join(path, 'koma', random.choice([
                'koma_dirty', 'koma_kinki', 'koma_kinki_torafu', 'koma_ryoko', 'koma_ryoko_torafu']))
        if source == 'sozai':
            path = os.path.join(path, 'japanese-chess', 'koma', '60x64')
        img = Image.open(os.path.join(path, '{}.png'.format(random.choice(self.pieceMap[source][piece]))))
        if source == 'irasutoya' and piece.startswith('W_'):
            img = img.rotate(180)
        return img


class ShineleckomaGenerator(Generator):
    def __init__(self, imageDir):
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
        self.bmp = []
        baseDir = os.path.join(imageDir, 'shineleckoma')
        for fileName in os.listdir(baseDir):
            self.bmp.append(os.path.normpath(os.path.join(baseDir, fileName)))
        if len(self.bmp) == 0:
            raise 'There are no images in shineleckoma'

    def generate(self, piece):
        bmp = random.choice(self.bmp)
        img = Image.open(bmp)
        board = Image.new('RGB', (463, 472), color=self.backgroundColor)
        draw = ImageDraw.Draw(board)
        # draw lines, points
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
        # draw piece
        file, rank = random.randrange(9), random.randrange(9)
        loc = self.pieceMap[piece]
        if loc is not None:
            piece = img.crop(box=(loc[0] * 43, loc[1] * 48, (loc[0] + 1) * 43, (loc[1] + 1) * 48))
            board.paste(piece, box=(
                file * 50 + 10, rank * 51 + 8,
                file * 50 + 53, rank * 51 + 56))
        # crop
        box = (
            file * 50 + 6 + 51 * random.normalvariate(0.0, 0.02),
            rank * 51 + 6 + 52 * random.normalvariate(0.0, 0.02),
            file * 50 + 57 + 51 * random.normalvariate(0.0, 0.02),
            rank * 51 + 58 + 52 * random.normalvariate(0.0, 0.02))
        cropped = board.crop(box=box)
        # resize
        add_noise(cropped)
        return cropped.resize((IMAGE_SIZE, IMAGE_SIZE), resample=random.choice(RESAMPLES))


class CharacterGenerator(Generator):
    def __init__(self, fonts):
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
            'BLANK': None,
        }

    def generate(self, piece):
        size = random.randrange(600, 1200)
        step = size / 10.0
        img = Image.new('RGB', (size, size), color='white')
        draw = ImageDraw.Draw(img)
        # draw lines
        for i in range(10):
            width = 2 if i == 0 or i == 9 else 1
            draw.line([(step * (i + 0.5), step * 0.5), (step * (i + 0.5), step * 9.5)], fill=0, width=width)
            draw.line([(step * 0.5, step * (i + 0.5)), (step * 9.5, step * (i + 0.5))], fill=0, width=width)
        # draw piece
        file, rank = random.randrange(9), random.randrange(9)
        if piece.startswith('B_') or piece.startswith('W_'):
            pieceSize = step * random.uniform(0.80, 0.95)
            pieceImg = Image.new('RGB', (int(pieceSize), int(pieceSize)), color='white')
            font = ImageFont.truetype(random.choice(self.fonts), size=int(pieceSize))
            pieceDraw = ImageDraw.Draw(pieceImg)
            s = random.choice(self.pieceMap[piece.split('_', 2)[1]])
            if s.startswith(u'成'):
                pieceDraw.text((0, 0), s[1], fill=0, font=font)
                nariImg = Image.new('RGB', (int(pieceSize), int(pieceSize) * 2), color='white')
                nariDraw = ImageDraw.Draw(nariImg)
                nariDraw.text((0, 0), u'成', fill=0, font=font)
                nariImg.paste(pieceImg, box=(0, int(pieceSize)))
                pieceImg = nariImg.resize((pieceImg.width, pieceImg.height), resample=random.choice(RESAMPLES))
            else:
                pieceDraw.text((0, 0), s, fill=0, font=font)
            if piece.startswith('W_'):
                pieceImg = pieceImg.rotate(180)
            img.paste(pieceImg, box=(
                int(step * (file + 1) - pieceSize * 0.5) + 1,
                int(step * (rank + 1) - pieceSize * 0.5) + 1))
        # crop
        box = (
            step * (file + 0.5) + step * random.normalvariate(0.0, 0.02),
            step * (rank + 0.5) + step * random.normalvariate(0.0, 0.02),
            step * (file + 1.5) + step * random.normalvariate(0.0, 0.02),
            step * (rank + 1.5) + step * random.normalvariate(0.0, 0.02))
        cropped = img.crop(box=box)
        # resize
        add_noise(cropped)
        return cropped.resize((IMAGE_SIZE, IMAGE_SIZE), resample=random.choice(RESAMPLES))
