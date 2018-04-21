#!/usr/bin/env python
import os
import urllib.request
from bs4 import BeautifulSoup


class Downloader:
    url = 'http://shineleckoma.web.fc2.com/'

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def run(self):
        for path in ['page1.htm', 'page2.htm', 'page3.htm', 'page4.htm']:
            self.downloadImages(path)

    def downloadImages(self, path):
        soup = BeautifulSoup(urllib.request.urlopen(Downloader.url + path), 'html.parser')
        for img in soup.find_all('img'):
            fileName = img['src']
            if not fileName.endswith('.bmp'):
                continue
            print('download {}...'.format(fileName))
            savePath = os.path.join(self.dir, fileName)
            imageUrl = Downloader.url + fileName
            urllib.request.urlretrieve(imageUrl, savePath)


if __name__ == '__main__':
    saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'shineleckoma')
    Downloader(saveDir).run()
