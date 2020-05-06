from helpers import TMP_DIR
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path('./data')
PUBMED_DIR = DATA_DIR / 'abstracts' / 'pubmed'
OUTPUT_FILE = PUBMED_DIR / 'pubmed_abstracts.txt'

headers = ['id', 'source', 'topic', 'title', 'doi', 'abstract']

sub_extra_xml_heads = re.compile(r'<\/PubmedArticleSet>\n?<\?xml version="1\.0" \?>\n?' +
    r'<\!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, .+?//EN"\s' +
    r'"https:\/\/dtd\.nlm\.nih\.gov/ncbi/pubmed/out/pubmed_[0-9]+?\.dtd">\n?' +
    r'<PubmedArticleSet>\n?').sub

def clean_xml_dump(xml_dump_file):
    xml_file = xml_dump_file.parent / xml_dump_file.stem
    print('Cleaning {}.'.format(xml_dump_file.name))
    with xml_dump_file.open('r') as fdin, xml_file.open('w', errors='replace', encoding='utf-8') as fdout:
        fdout.write(sub_extra_xml_heads('', fdin.read()).replace(' � ', ''))

def get_topic_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    instance_tmpl = {header: None for header in headers}
    instance_tmpl['source'] = 'pubmed'
    instance_tmpl['topic'] = xml_file.stem

    instances = []
    count = 0

    for child in root.iter('PubmedArticle'):
        count+=1
        instance = instance_tmpl.copy()
        article_data = child.find('MedlineCitation').find('Article')
        pubmed_data = child.find('PubmedData')

        try:
            article_id_data_list = pubmed_data.find('ArticleIdList').findall('ArticleId')
            
            for article_id_data in article_id_data_list:
                id_type = article_id_data.get('IdType')
                if id_type == 'pubmed':
                    instance['id'] = ' '.join(list(article_id_data.itertext())).strip()
                elif id_type == 'pmc' and instance['id'] == None:
                    instance['id'] = ' '.join(list(article_id_data.itertext())).strip()
                elif id_type == 'doi':
                    instance['doi'] = 'https://doi.org/' + ' '.join(list(article_id_data.itertext())).strip()
            
            instance['title'] = ' '.join(list(article_data.find('ArticleTitle').itertext())).strip().replace('\t', ' ').replace('\n', ' ')
            instance['abstract'] = ' '.join(list(article_data.find('Abstract').find('AbstractText').itertext())).strip().replace('\t', ' ').replace('\n', ' ')
            if not instance['abstract']:
                continue
        except Exception as e:
            continue


        instances.append(instance)
    
    print("Total instances:", count)
    return instances


def to_tsv(data):
    for instance in data:
        try:
            fd.write('\t'.join([str(instance.get(header) if not None else '') for header in headers]) + '\n')
        except Exception as e:
            print(e)
            print(instance)


if __name__ == "__main__":

    xml_dump_files = list(PUBMED_DIR.glob('*.xml.dump'))
    
    for xml_dump_file in xml_dump_files:
        clean_xml_dump(xml_dump_file)

    xml_files = list(PUBMED_DIR.glob('*.xml'))

    with OUTPUT_FILE.open('w', errors='replace') as fd:
        fd.write('\t'.join(headers) + '\n')
        for xml_file in xml_files:
            print("Parsing", xml_file.stem)
            data = get_topic_data(xml_file)
            print("Available instances:", len(data))
            to_tsv(data)
