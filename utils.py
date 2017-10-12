import requests
from bs4 import BeautifulSoup
from random import shuffle

first_letters_of_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł',
    'M', 'N', 'O', 'P', 'R', 'S', 'Ś', 'T', 'U', 'W', 'Z', 'Ż',
]

data_path = 'data/polish_names.txt'


def link_to_name(l):
    raw = str(l.get('title'))
    return bytes(raw, "utf-8").decode('unicode-escape').replace('"', '').lower()


def download_names():
    base_url = "https://pl.wikipedia.org/w/api.php?format=json&action=parse"
    file = open(data_path, 'w')
    for letter in first_letters_of_names:
        url = "&page=Imiona_na_liter%C4%99_{0}".format(letter)
        cont = requests.get(base_url + url).content
        soup = BeautifulSoup(cont, 'html.parser')
        name_links = soup.find_all('ul')[0].find_all("a")
        for link in name_links:
            l = link_to_name(link)
            file.write(l + '\n')
    file.close()

def shuffle_names():
    with open(data_path) as f:
        lines = f.read().splitlines()
        shuffle(lines)
    file = open(data_path, 'w')
    for l in lines:
        file.write(l + '\n')
    file.close()



