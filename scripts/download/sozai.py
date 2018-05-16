#!/usr/bin/env python
import os
import tempfile
import zipfile
import urllib.request


class Downloader:
    url = 'http://sozai.7gates.net/img/download/japanese-chess140523.zip'

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def run(self):
        tempFile = os.path.join(tempfile.gettempdir(), 'japanese-chess140523.zip')
        print('download {}...'.format(tempFile))
        urllib.request.urlretrieve(Downloader.url, tempFile)
        with zipfile.ZipFile(tempFile) as z:
            z.extractall(self.dir)


if __name__ == '__main__':
    saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sozai')
    Downloader(saveDir).run()
