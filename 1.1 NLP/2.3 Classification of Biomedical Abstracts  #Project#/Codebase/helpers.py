from pathlib import Path
import pickle
import time
from datetime import datetime
from pytz import timezone

TZ = 'America/Chicago'

DATA_DIR = Path('./data')
REPORT_DIR = DATA_DIR / 'reports'
ABSTRACT_DIR = DATA_DIR / 'abstracts'
ABSTRACT_BIORXIV_DIR = ABSTRACT_DIR / 'biorxiv'
ABSTRACT_PUBMED_DIR = ABSTRACT_DIR / 'pubmed'
MISC_DIR = DATA_DIR / 'misc'

OUTPUT_DIR = Path('./output')
PLOT_DIR = OUTPUT_DIR / 'plots'
PICKLE_DIR = OUTPUT_DIR / 'pickled'
TSV_DIR = OUTPUT_DIR / 'tsv'
TMP_DIR = OUTPUT_DIR / 'tmp'

class Logger():
    def __init__(self, no_date=True):
        self.no_date = no_date
        self._start_time = time.time()
        self._prev_time = self._start_time
        self._total_delta_time = 0
        self._delta_time = self._total_delta_time

    def plog(self, *args):
        current_time = time.time()

        dt = timezone(TZ).localize(datetime.now())
        dt = dt.strftime('%H:%M:%S') if self.no_date else dt.strftime('%Y-%m-%d %H:%M:%S')

        self._delta_time = current_time - self._prev_time
        self._total_delta_time = current_time - self._start_time
        self._prev_time = current_time
        
        print('<{}> ::'.format(dt), *args, ' -- Time since last: {:.02f} sec. ; Total Time: {:.02f} sec.'.format(self._delta_time, self._total_delta_time))

def pickle_object(obj, filename):
    filepath = PICKLE_DIR / filename
    try:
        with filepath.open('wb') as fd:
            pickle.dump(obj, fd)
        return True
    except Exception as e:
        print("Could not pickle data to {}.".format(filepath)) 
        print(e)
        return False

def unpickle_object(filename):
    filepath = PICKLE_DIR / filename
    try:
        with filepath.open('rb') as fd:
            return pickle.load(fd)
    except Exception as e:
        print("Could not unpickle data from {}.".format(filepath)) 
        print(e)
        return False
