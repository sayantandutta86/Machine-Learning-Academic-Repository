import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

import numpy as np
import os
import nltk
import itertools
import io

'''Each review is contained in a separate .txt file and organized into separate directories depending on if they’re positive/negative or part of the train/test/unlabeled splits. It’s easier to re-work the data before training any models for multiple reasons.

It’s faster to combine everything into fewer files.
We need to tokenize the dataset (split long strings up into individual words/symbols).
We need to know how many unique tokens there are to decide on our vocabulary/model size.
We don’t want to constantly be converting strings to token IDs within the training loop (takes non-negligible amount of time).
Storing everything as token IDs directly leads to the 1-hot representation necessary for input into our model.'''

path = "/u/training/tra272/scratch/hw5"
os.chdir(path)

## create directory to store preprocessed data
if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')

## get all of the training reviews (including unlabeled reviews)
train_directory = '/projects/training/bayw/NLP/aclImdb/train/'

pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')

pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]
unsup_filenames = [train_directory+'unsup/'+filename for filename in unsup_filenames]

filenames = pos_filenames + neg_filenames + unsup_filenames

count = 0
x_train = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_train.append(line)
    count += 1
    print(count)
	
'''The top part of the code simply results in a list of every text file containing a movie review in the variable filenames. The first 12500 are positive reviews, the next 12500 are negative reviews, and the remaining are unlabeled reviews. For each review, we remove text we don’t want (’
’ and ‘\x96’), tokenize the string using the nltk package, and make everything lowercase. Here is an example movie review before and after tokenization:

"`Mad Dog' Earle is back, along with his sad-sack moll Marie, and that fickle clubfoot Velma. So are Babe and Red, Doc and Big Mac, and even the scenery-chewing mutt Pard. The only thing missing is a good reason for remaking Raoul Walsh's High Sierra 14 years later without rethinking a line or a frame, and doing so with talent noticeably a rung or two down the ladder from that in the original. (Instead of Walsh we get Stuart Heisler, for Humphrey Bogart we get Jack Palance, for Ida Lupino Shelley Winters, and so on down through the credits.) The only change is that, this time, instead of black-and-white, it's in Warnercolor; sadly, there are those who would count this an improvement.<br /><br />I Died A Thousand Times may be unnecessary \x96 and inferior \x96 but at least it's not a travesty; the story still works on its own stagy terms. Earle (Palance), fresh out of the pen near Chicago, drives west to spearhead a big job masterminded by ailing kingpin Lon Chaney, Jr. \x96 knocking over a post mountain resort. En route, he almost collides with a family of Oakies, when he's smitten with their granddaughter; the smiting holds even when he discovers she's lame. Arriving at the cabins where the rest of gang holes up, he finds amateurish hotheads at one another's throats as well as Winters, who throws herself at him (as does the pooch). Biding time until they get a call from their inside man at the hotel, Palance (to Winter's chagrin) offers to pay for an operation to cure the girl's deformity, a gesture that backfires. Then, the surgical strike against the resort turns into a bloodbath. On the lam, Palance moves higher into the cold Sierras....<br /><br />It's an absorbing enough story, competently executed, that lacks the distinctiveness Walsh and his cast brought to it in 1941, the year Bogie, with this role and that of Sam Spade in the Maltese Falcon, became a star. And one last, heretical note: Those mountains do look gorgeous in color.<br /><br />"


['`', 'mad', 'dog', "'", 'earle', 'is', 'back', ',', 'along', 'with', 'his', 'sad-sack', 'moll', 'marie', ',', 'and', 'that', 'fickle', 'clubfoot', 'velma', '.', 'so', 'are', 'babe', 'and', 'red', ',', 'doc', 'and', 'big', 'mac', ',', 'and', 'even', 'the', 'scenery-chewing', 'mutt', 'pard', '.', 'the', 'only', 'thing', 'missing', 'is', 'a', 'good', 'reason', 'for', 'remaking', 'raoul', 'walsh', "'s", 'high', 'sierra', '14', 'years', 'later', 'without', 'rethinking', 'a', 'line', 'or', 'a', 'frame', ',', 'and', 'doing', 'so', 'with', 'talent', 'noticeably', 'a', 'rung', 'or', 'two', 'down', 'the', 'ladder', 'from', 'that', 'in', 'the', 'original', '.', '(', 'instead', 'of', 'walsh', 'we', 'get', 'stuart', 'heisler', ',', 'for', 'humphrey', 'bogart', 'we', 'get', 'jack', 'palance', ',', 'for', 'ida', 'lupino', 'shelley', 'winters', ',', 'and', 'so', 'on', 'down', 'through', 'the', 'credits', '.', ')', 'the', 'only', 'change', 'is', 'that', ',', 'this', 'time', ',', 'instead', 'of', 'black-and-white', ',', 'it', "'s", 'in', 'warnercolor', ';', 'sadly', ',', 'there', 'are', 'those', 'who', 'would', 'count', 'this', 'an', 'improvement', '.', 'i', 'died', 'a', 'thousand', 'times', 'may', 'be', 'unnecessary', 'and', 'inferior', 'but', 'at', 'least', 'it', "'s", 'not', 'a', 'travesty', ';', 'the', 'story', 'still', 'works', 'on', 'its', 'own', 'stagy', 'terms', '.', 'earle', '(', 'palance', ')', ',', 'fresh', 'out', 'of', 'the', 'pen', 'near', 'chicago', ',', 'drives', 'west', 'to', 'spearhead', 'a', 'big', 'job', 'masterminded', 'by', 'ailing', 'kingpin', 'lon', 'chaney', ',', 'jr.', 'knocking', 'over', 'a', 'post', 'mountain', 'resort', '.', 'en', 'route', ',', 'he', 'almost', 'collides', 'with', 'a', 'family', 'of', 'oakies', ',', 'when', 'he', "'s", 'smitten', 'with', 'their', 'granddaughter', ';', 'the', 'smiting', 'holds', 'even', 'when', 'he', 'discovers', 'she', "'s", 'lame', '.', 'arriving', 'at', 'the', 'cabins', 'where', 'the', 'rest', 'of', 'gang', 'holes', 'up', ',', 'he', 'finds', 'amateurish', 'hotheads', 'at', 'one', 'another', "'s", 'throats', 'as', 'well', 'as', 'winters', ',', 'who', 'throws', 'herself', 'at', 'him', '(', 'as', 'does', 'the', 'pooch', ')', '.', 'biding', 'time', 'until', 'they', 'get', 'a', 'call', 'from', 'their', 'inside', 'man', 'at', 'the', 'hotel', ',', 'palance', '(', 'to', 'winter', "'s", 'chagrin', ')', 'offers', 'to', 'pay', 'for', 'an', 'operation', 'to', 'cure', 'the', 'girl', "'s", 'deformity', ',', 'a', 'gesture', 'that', 'backfires', '.', 'then', ',', 'the', 'surgical', 'strike', 'against', 'the', 'resort', 'turns', 'into', 'a', 'bloodbath', '.', 'on', 'the', 'lam', ',', 'palance', 'moves', 'higher', 'into', 'the', 'cold', 'sierras', '...', '.', 'it', "'s", 'an', 'absorbing', 'enough', 'story', ',', 'competently', 'executed', ',', 'that', 'lacks', 'the', 'distinctiveness', 'walsh', 'and', 'his', 'cast', 'brought', 'to', 'it', 'in', '1941', ',', 'the', 'year', 'bogie', ',', 'with', 'this', 'role', 'and', 'that', 'of', 'sam', 'spade', 'in', 'the', 'maltese', 'falcon', ',', 'became', 'a', 'star', '.', 'and', 'one', 'last', ',', 'heretical', 'note', ':', 'those', 'mountains', 'do', 'look', 'gorgeous', 'in', 'color', '.']

Notice how symbols like periods, parenthesis, etc. become their own tokens and it splits up contractions (“it’s” becomes “it” and “’s”). It’s not perfect and can be ruined by typos or lack of punctuation but works for the most part. We now have a list of tokens for every review in the training dataset in the variable x_train. We can do the same thing for the test dataset and the variable x_test.

'''

## get all of the test reviews
test_directory = '/projects/training/bayw/NLP/aclImdb/test/'

pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')

pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]

filenames = pos_filenames+neg_filenames

count = 0
x_test = []
for filename in filenames:
    with io.open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />',' ')
    line = line.replace('\x96',' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]

    x_test.append(line)
    count += 1
    print(count)

## number of tokens per review
no_of_tokens = []
for tokens in x_train:
    no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))


'''Total: 20091975 Min: 10 Max: 2859 Mean: 267.893 Std: 198.5532807191393

The mean review contains ~267 tokens with a standard deviation of ~200. Although there are over 20 million total tokens, they’re obviously not all unique. We now want to build our dictionary/vocabulary.'''

### word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

'''
We have two more variables: word_to_id and id_to_word. You can access id_to_word[index] to find out the token for a given index or access word_to_id[token] to find out the index for a given token. They’ll both be used later on. We can also see there are roughly 200,000 unique tokens with len(id_to_word). Realistically, we don’t want to use all 200k unique tokens. An embedding layer with 200k tokens and 500 outputs has 100 million weights which is far too many considering the training dataset only has 20 million tokens total. Additionally, the tokenizer sometimes creates a unique token that is actually just a combination of other tokens because of typos and other odd things. Here are a few examples from id_to_word (“majesty.these”, “producer/director/star”, “1¢”)

We should organize the tokens by their frequency in the dataset to give us a better idea of choosing our vocabulary size.
'''


## let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])
	
'''
Notice we used word_to_id to convert all of the tokens for each review into a sequence of token IDs instead (this is ultimately what we are going for but we want the IDs to be in a different order). We then accumulate the total number of occurrences for each word in the array count. Finally, this is used to sort the vocabulary list. We are left with id_to_word in order from most frequent tokens to the least frequent, word_to_id gives us the index for each token, and count simply contains the number of occurrences for each token.

(array([163890,  26981,   7822,   1367]), array([    1,    10,   100,  1000, 10000]))
the 1009387.0
, 829575.0
. 819080.0
and 491830.0
a 488491.0
of 438545.0
to 405661.0
is 330904.0
it 285715.0
in 280618.0

'''

'''The histogram output gives us a better understanding of the actual dataset. Over 80% (~160k) of the unique tokens occur between 1 and 10 times while only ~5% occur more than 100 times each. Using np.sum(count[0:100]) tells us over half of all of the 20 million tokens are the most common 100 words and np.sum(count[0:8000]) tells us almost 95% of the dataset is contained within the most common 8000 words.'''

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

## save dictionary
np.save('preprocessed_data/imdb_dictionary.npy',np.asarray(id_to_word))

## save training data to single text file
with io.open('preprocessed_data/imdb_train.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

## save test data to single text file
with io.open('preprocessed_data/imdb_test.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

glove_filename = '/projects/training/bayw/NLP/glove.840B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

'''Here is where we convert everything to the exact format we want for training purposes. Notice the test dataset may have unique tokens our model has never seen before. We can anticipate this ahead of time by actually reserving index 0 for an unknown token. This is why I assign a -1 if the token isn’t part of word_to_id and add +1 to every id. Just be aware of this that id_to_word is now off by 1 index if you actually want to convert ids to words.

We could also decide later on if we only want to use a vocabulary size of 8000 for training and just assign any other token ID in the training data to 0. This way it can develop its own embedding for unknown tokens which can help out when it inevitably sees unknown tokens during testing.'''

np.save('preprocessed_data/glove_dictionary.npy',glove_dictionary)
np.save('preprocessed_data/glove_embeddings.npy',glove_embeddings)

with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open('preprocessed_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

