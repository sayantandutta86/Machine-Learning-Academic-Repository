from bs4 import BeautifulSoup as BS
from pathlib import Path
import urllib.parse as urlparse
import json

# from selenium import webdriver
# from selenium.webdriver.common.action_chains import ActionChains

import requests
import time
import random

GECKOPATH = 'C:/Program Files/Mozilla Firefox/geckodriver-v0.24.0-win64'
# DRIVER = webdriver.Firefox(GECKOPATH)
BASE_URL = 'https://biorxiv.org'
COLLECTION_URL = BASE_URL + '/collection'
ABSTRACT_URL = BASE_URL + '/highwire/article_citation_preview'

HEADERS = {
    'from': 'jtomasz2@illinois.edu',
}

HTML_BAK = Path('./data/html_bak')

RANDOM_INTERVAL = [0.34, 1.1]

SUBJECTS = [
    "bioinformatics",
    "neuroscience",
    "microbiology",
    "pharmacology and toxicology",
    "epidemiology",
    "genomics",
    "immunology"
]

def get_page_number(url):
    parsed_url = urlparse.urlparse(url)
    query_str = urlparse.parse_qs(parsed_url.query)
    try:
        return int(query_str.get('page')[0])
    except Exception as e:
        print(e)
        return 0


def save_article_data(article_section, abstract_output_dir):
    article_id = article_section.attrs.get('data-node-nid')
    abstract_file = (abstract_output_dir / '{}.json'.format(article_id))

    if abstract_file.exists():
        return
    
    abstract_res = requests.get('{}/{}'.format(ABSTRACT_URL, article_id), headers=HEADERS)
    abstract_html = BS(abstract_res.text, 'lxml')

    article_data = {
        'title': article_section.find('span', 'highwire-cite-title').text.strip(),
        'biorxiv_id': article_id,
        'doi': article_section.find('span', 'highwire-cite-metadata-doi').text[4:].strip(),
        'abstract': abstract_html.text
    }

    with abstract_file.open('w', errors='xmlcharrefreplace') as fp:
        json.dump(article_data, fp)

    time.sleep(random.uniform(*RANDOM_INTERVAL))


def get_html(url, backup_file):
    if not backup_file.exists():
        res = requests.get(url, headers=HEADERS)
        with backup_file.open('w', errors='xmlcharrefreplace') as fp:
            fp.write(res.text)
        return BS(res.text, 'lxml')
    else:
        with backup_file.open('r', errors='xmlcharrefreplace') as fp:
            return BS(fp.read(), 'lxml')


def get_abstracts_at_url(url, subject):
    page_number = get_page_number(url)
    print('Reading page {:d} of the subject {}.'.format(page_number, subject))
    abstract_output_dir = Path('./data/abstracts/biorxiv/{}'.format(subject))

    if not abstract_output_dir.exists():
        try:
            abstract_output_dir.mkdir(parents=True)
        except FileExistsError as e:
            print(abstract_output_dir, "already exists.")
    
    soup = get_html(url, (HTML_BAK / '{}_{:03d}.html'.format(subject, page_number)))
    
    article_sections = soup.find_all('div', class_='highwire-article-citation')
    for article_section in article_sections:
        save_article_data(article_section, abstract_output_dir)

    next_page = soup.find('a', 'link-icon-after')
    if next_page:
        next_url = '{}{}'.format(BASE_URL, next_page.attrs.get('href'))
        time.sleep(random.uniform(*RANDOM_INTERVAL))
        get_abstracts_at_url(next_url, subject)



if __name__ == '__main__':
    for subject in SUBJECTS[2:]:
        try:
            url = '{}/{}?page=0'.format(COLLECTION_URL, subject)
            get_abstracts_at_url(url, subject)
        except Exception as e:
            print(e)

