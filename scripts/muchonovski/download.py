import os
import urllib.request
from bs4 import BeautifulSoup


class Downloader:
    def __init__(self, rootUrl, directory):
        saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', directory)
        os.makedirs(saveDir, exist_ok=True)
        self.rootUrl = rootUrl
        self.dir = saveDir

    def run(self):
        for path in ['koma.html', 'ban.html', 'masu.html']:
            self.downloadImages(path)

    def downloadImages(self, path):
        soup = BeautifulSoup(urllib.request.urlopen(self.rootUrl + path), 'html.parser')
        for img in soup.find_all('img'):
            if img['src'].startswith('other'):
                continue
            print('download {}...'.format(img['src']))
            dirname, fileName = os.path.split(img['src'])
            saveDir = os.path.join(self.dir, *dirname.split('/'))
            os.makedirs(saveDir, exist_ok=True)
            savePath = os.path.join(saveDir, fileName)
            urllib.request.urlretrieve(self.rootUrl + img['src'], savePath)


if __name__ == '__main__':
    Downloader('http://mucho.girly.jp/bona/', 'muchonovski').run()
