#!/usr/bin/env python
import argparse
import os
import random
import re
from generators import AlphaGenerator, ShineleckomaGenerator, CharacterGenerator

PIECES = [
    'B_FU', 'W_FU',
    'B_TO', 'W_TO',
    'B_KY', 'W_KY',
    'B_NY', 'W_NY',
    'B_KE', 'W_KE',
    'B_NK', 'W_NK',
    'B_GI', 'W_GI',
    'B_NG', 'W_NG',
    'B_KI', 'W_KI',
    'B_KA', 'W_KA',
    'B_UM', 'W_UM',
    'B_HI', 'W_HI',
    'B_RY', 'W_RY',
    'B_OU', 'W_OU',
    'BLANK'
]


def run(dataDir, g):
    generators, weights = [], []
    for generator, weight in g.items():
        generators.append(generator)
        weights.append(weight)

    for piece in PIECES:
        for i in range(2000):
            fileName = '{:04d}.jpg'.format(i)
            savePath = os.path.join(dataDir, piece, fileName)
            generator = random.choices(generators, weights=weights)[0]
            img = generator.generate(piece)
            print('{}: generated {}...'.format(piece, fileName))
            with open(savePath, 'w') as fp:
                img.save(fp, quality=random.randint(80, 100))
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fonts', nargs='*', required=True)
    args = parser.parse_args()
    fontFilter = re.compile(r'W[6-9]')
    fonts = [f for f in args.fonts if not fontFilter.search(f)]

    dataDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset')
    imageDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'images')
    generators = {
        AlphaGenerator(imageDir): 5,
        ShineleckomaGenerator(imageDir): 4,
        CharacterGenerator(fonts): 1,
    }
    run(dataDir, generators)
