"""
    Connect and download XML from PubMed with Biopython
"""
import helpers
from Bio import Entrez
import pickle
from pathlib import Path
import datetime
import logging
import json
import time
import re
from xml.dom import minidom
from xml.etree import ElementTree as ET

from collections import OrderedDict
from urllib.error import HTTPError
from pprint import PrettyPrinter
pprint = PrettyPrinter().pprint

from helpers import TMP_DIR

BIORXIV_PATH = Path('./data/abstracts/biorxiv')
DST_PATH = Path('./data/abstracts/pubmed')
if not DST_PATH.exists():
    DST_PATH.mkdir(parents=True)

PM_DATABASE = 'pubmed'

# Biorxiv subject will be translated a synonomous [MeSH Major Topic]
SUBJECTS = [
    "bioinformatics",
    "neuroscience",
    "microbiology",
    "pharmacology and toxicology",
    "epidemiology",
    "genomics",
    "immunology"
]

Entrez.email = 'jtomasz2@illinois.edu'


def query(terms):
    """Query pubmed."""

    query_args = {
        'term': terms,
    }

    handle = Entrez.egquery(**query_args)
    result = Entrez.read(handle).get('eGQueryResult')
    handle.close()

    if result:
        result = [v for v in result
                    if v.get('DbName') == 'pubmed' and
                    v.get('Status') == 'Ok']
        return result[0] if result else []
    else:
        print('Query returned no results.')
        return

def search(terms=None, total_count=None, sort='relevance', retmode='xml',
            usehistory=False, retmax=100, retstart=0, **kwargs):
    """
    Wrap 'esearch' method for directed usage.

    The necessary email parameter added to the request.
    """

    search_results = {}

    search_args = {
        'db': PM_DATABASE,
        'term': terms,
        'retmode': retmode,
        'retmax': retmax,
        'sort': sort
    }
    search_args.update(**kwargs)

    search_args.update({
        'usehistory': 'y'
    })

    history_dict = {}

    # from http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc135
    
    if not total_count:
        total_count = retmax

    for start in range(retstart, total_count, retmax):
        if (all([k in search_results.keys()
            for k in ['WebEnv', 'QueryKey']]) and
                not search_args.get('webenv')):
            search_args.update({
                'webenv': search_results['WebEnv'],
                'query_key': search_results['QueryKey'],
            })

        end = min(total_count, start + retmax)
        print('Downloading search results {} to {}.'.format(
                start + 1, end))

        search_handle = None

        # request may occationally fail, 3 tries should be enough
        for attempt in range(3):
            if not search_handle:
                try:
                    search_args['retstart'] = start
                    search_handle = Entrez.esearch(**search_args)
                except HTTPError as e:
                    if 500 <= e.code <= 599:
                        print("Received error: {}".format(e))
                        print("Attempt {} of 3".format(attempt))
                        time.sleep(15)  # wait 15 seconds
                    else:
                        raise(e)
        results = Entrez.read(search_handle)
        search_handle.close()

        if search_results:
            search_results['IdList'].extend(results['IdList'])
            search_results['RetCount'] += \
                int(results['RetMax'])
        else:
            search_results = results
            search_results['RetStart'] = \
                int(search_results['RetStart'])
            search_results['RetCount'] = \
                int(search_results['RetMax'])
        time.sleep(1)  # wait 1 second

    print('Search Complete. Retrieved {} results.'.format(
            len(search_results.get('IdList'))))

    return search_results


def fetch(fp, article_ids, total_count=None, history_dict=None, retmode='xml',
            retmax=100, retstart=0, db=None, **kwargs):
    """Request article metadata with the 'efetch' method."""

    fetch_results = {}

    fetch_args = {
        'db': PM_DATABASE,
        'retmode': retmode,
        'retmax': retmax
    }
    fetch_args.update(**kwargs)
    fetch_args.update({'id': article_ids})

    # from http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc135

    if history_dict:
        fetch_args.update({
            **history_dict,
            'idtype': 'acc'  # Accession numbers
        })

    if not total_count:
        total_count = retmax

    print("Fetch count set to {}.".format(total_count))

    for start in range(retstart, total_count, retmax):

        end = min(total_count, start + retmax)
        print('Downloading fetch results records {} to {}.'.format(
                start + 1, end))
        # request may occationally fail, 3 tries should be enough
        for attempt in range(3):
            try:
                fetch_args['retstart'] = start
                fetch_handle = Entrez.efetch(**fetch_args)
            except HTTPError as e:
                if 400 <= e.code <= 599:
                    print("Received error: {}".format(e))
                    print("Attempt {} of 3".format(attempt))
                    time.sleep(15)  # wait 15 seconds
                else:
                    raise(e)
            except Exception as e:
                print("Non-HTTP error occured.")
                raise(e)

        # results = Entrez.read(fetch_handle, validate=False)
        results = fetch_handle.read()

        if fp:
            fp.write(results)
            continue

        if not results:
            fetch_handle.close()
            print("Failed to retrieve more results.")
            break


        fetch_handle.close()
        time.sleep(1)  # wait 1 second


 
if __name__ == "__main__":
    # Biorxiv only goes back to 2013. Limit pubmed to the same
    MINDATE = 2013
    for subject in SUBJECTS[2:]:
        biorxiv_subject_path = (BIORXIV_PATH / '{}'.format(subject))
        search_term = '{}[MeSH Major Topic]'.format(subject)        

        # query_res = query(search_term) 
        # article_count = int(query_res['Count'])
        article_count = len(list(biorxiv_subject_path.glob('*.json'))) * 2
        print('Count of {} articles: {:d}'.format(subject, article_count))
        
        search_res = search(search_term, article_count, retmax=1000, mindate=MINDATE)
        article_ids = search_res.get('IdList')
        history_dict = {
            'webenv': search_res.get('WebEnv'),
            'query_key': search_res.get('QueryKey')
        }

        with (TMP_DIR / '{}.tmp'.format(subject)).open('w') as fd:
            fd.write('\n'.join(article_ids))

        with (DST_PATH / '{}.xml.dump'.format(subject)).open('w', errors='xmlcharrefreplace') as fp:
            fetch_res = fetch(fp, article_ids, article_count, history_dict, retmax=500, mindate=MINDATE)


# def link(self, article_ids=None, retmode='text', usehistory=False,
#             retmax=100, retstart=0, db=None, dbto=None, **kwargs):
#     """Request article metadata with the 'elink' method."""
#     article_ids = article_ids or self._results['search'].get('IdList')
#     if not article_ids and not usehistory:
#         raise Exception("The 'get_links' method requires article ids. " +
#                         "Please supply article ids or " +
#                         "call 'search(...)' prior to this method.")

#     self._results['links'] = {}

#     link_args = {
#         'dbfrom': db or self.db,
#         'db': dbto or self.dbto,
#         'retmode': retmode,
#         'retmax': retmax,
#     }
#     link_args.update(**kwargs)
#     link_args.update({'id': article_ids})

#     print(("Retrieving {} links from the {} database " +
#             "to the {} database.").format(len(article_ids),
#                                             db or self.db, dbto or self.dbto))

#     backupfile = (BACKUP_PATH / './{}_links_backup.json'.format(
#         self._init_datetime)).open('a')

#     link_handle = Entrez.elink(**link_args)
#     results = Entrez.read(link_handle)
#     link_handle.close()

#     missing_count = 0

#     for res in results:
#         pmid = res.get("IdList")[0]
#         pmcid = None
#         try:
#             pmcid = res["LinkSetDb"][0]["Link"][0]["Id"]
#         except Exception as e:
#             print(pmid, 'not found')
#             missing_count += 1
#             pass

#         self._results['links'][pmid] = pmcid

#     print("Matches not found for {} records.".format(missing_count))
#     json.dump(self._results['links'], backupfile)
#     backupfile.close()

#     return self._results['links']



