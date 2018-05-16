#!/usr/bin/env python
import os
import urllib.request
from bs4 import BeautifulSoup


class Downloader:
    url = 'http://mucho.girly.jp/bona/'

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def run(self):
        for path in ['koma.html', 'ban.html', 'masu.html']:
            self.downloadImages(path)

    def downloadImages(self, path):
        soup = BeautifulSoup(urllib.request.urlopen(Downloader.url + path), 'html.parser')
        for img in soup.find_all('img'):
            if img['src'].startswith('other'):
                continue
            print('download {}...'.format(img['src']))
            dirName, fileName = img['src'].rsplit('/', 1)
            saveDir = os.path.join(self.dir, *dirName.split('/'))
            os.makedirs(saveDir, exist_ok=True)
            savePath = os.path.join(saveDir, fileName)
            imageUrl = Downloader.url + img['src']
            urllib.request.urlretrieve(imageUrl, savePath)


if __name__ == '__main__':
    saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'muchonovski')
    Downloader(saveDir).run()
