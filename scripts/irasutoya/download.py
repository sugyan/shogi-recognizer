import os
import urllib.request
from bs4 import BeautifulSoup


class Downloader:
    url = 'https://www.irasutoya.com/'

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def run(self):
        for path in ['2014/08/blog-post_52.html', '2014/08/blog-post_27.html']:
            self.downloadImages(path)

    def downloadImages(self, path):
        soup = BeautifulSoup(urllib.request.urlopen(Downloader.url + path), 'html.parser')
        for img in soup.find('div', class_='entry').find_all('img'):
            if 'thumbnail' in img['src']:
                continue
            print('download {}...'.format(img['src']))
            fileName = img['src'].split('/')[-1]
            savePath = os.path.join(self.dir, fileName)
            imageUrl = img['src']
            if imageUrl.startswith('//'):
                imageUrl = 'https:' + imageUrl
            urllib.request.urlretrieve(imageUrl, savePath)


if __name__ == '__main__':
    saveDir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'irasutoya')
    Downloader(saveDir).run()
