#!/usr/bin/env python
import os
import urllib.request


class Downloader:
    url = 'http://shineleckoma.web.fc2.com/'

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def run(self):
        fileNames = [
            'koma9.bmp',   'koma18.bmp',  'koma20.bmp',  'koma25.bmp',
            'koma35.bmp',  'koma36.bmp',  'koma40.bmp',  'koma41.bmp',
            'koma44.bmp',  'koma48.bmp',  'koma50.bmp',  'koma52.bmp',
            'koma56.bmp',  'koma59.bmp',  'koma61.bmp',  'koma63.bmp',
            'koma64.bmp',  'koma66.bmp',  'koma73.bmp',  'koma76.bmp',
            'koma82.bmp',  'koma86.bmp',  'koma92.bmp',  'koma94.bmp',
            'koma96.bmp',  'koma98.bmp',  'koma100.bmp', 'koma101.bmp',
            'koma105.bmp', 'koma107.bmp', 'koma109.bmp', 'koma113.bmp',
            'koma115.bmp', 'koma117.bmp', 'koma119.bmp', 'koma120.bmp',
            'koma121.bmp',
        ]
        for fileName in fileNames:
            self.downloadImages(fileName)

    def downloadImages(self, fileName):
        print('download {}...'.format(fileName))
        savePath = os.path.join(self.dir, fileName)
        imageUrl = Downloader.url + fileName
        urllib.request.urlretrieve(imageUrl, savePath)


if __name__ == '__main__':
    saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'shineleckoma')
    Downloader(saveDir).run()
