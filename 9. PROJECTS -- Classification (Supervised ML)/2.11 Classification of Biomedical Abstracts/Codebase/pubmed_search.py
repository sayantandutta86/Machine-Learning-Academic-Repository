import time

def search(search_terms=None, fields=None, sort='relevance', retmode='xml',
            usehistory=False, retmax=100, retstart=0, **kwargs):
    """
    Wrap 'esearch' method for directed usage.

    The necessary email parameter added to the request.
    """

    search_results = {}

    search_args = {
        'db': self.db,
        'term': terms,
        'field': fields,
        'retmode': retmode,
        'retmax': retmax,
        'sort': sort
    }
    search_args.update(**kwargs)

    if not self._results['query'].get('Count'):
        raise Exception("Call 'query' before making " +
                        "a search while using history.")

    search_args.update({
        'usehistory': 'y'
    })

    # from http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc135
    
    total_count = int(self._results['query'].get('Count'))


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