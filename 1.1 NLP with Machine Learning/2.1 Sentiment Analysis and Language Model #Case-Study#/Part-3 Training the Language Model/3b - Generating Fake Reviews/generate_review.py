# Author - Sayantan Dutta
# Program: generate_review.py
# Assignment- 3b

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist

import h5py
import time
import os
import io

import sys

'''Although a general language model assigns a probability P(w_0, w_1, …, w_n) over the entire sequence, we’ve actually trained ours to predict P(w_n|w_0, …, w_n-1) where each output is conditioned only on previous inputs. This gives us the ability to sample from P(w_n|w_0, …, w_n-1), feed this sampled token back into the model, and repeat this process in order to generate fake movie reviews. '''

from RNN_language_model import RNN_language_model

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}

'''We will actually utilize the vocabulary we constructed earlier to convert sampled indices back to their corresponding words.'''

model = torch.load('language.model')
print('model loaded...')
model.cuda()

model.eval()

## create partial sentences to "prime" the model
## this implementation requires the partial sentences
## to be the same length if doing more than one
tokens = [['i','love','this','movie','.'],['i','hate','this','movie','.']]
#tokens = [['a'],['i']]

token_ids = np.asarray([[word_to_id.get(token,-1)+1 for token in x] for x in tokens])

## preload phrase
x = Variable(torch.LongTensor(token_ids)).cuda()

embed = model.embedding(x) # batch_size, time_steps, features

state_size = [embed.shape[0],embed.shape[2]] # batch_size, features
no_of_timesteps = embed.shape[1]

model.reset_state()

outputs = []
for i in range(no_of_timesteps):

    h = model.lstm1(embed[:,0,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    h = model.decoder(h)

    outputs.append(h)

outputs = torch.stack(outputs)
outputs = outputs.permute(1,2,0)
output = outputs[:,:,-1]

'''We can start sampling at the very start or after the model has processed a few words already. The latter is akin to autocomplete. In this example, I’m generating two reviews. The first starts simply with the letter/word ‘a’ and the second starts with the letter/word ‘i’. These are both stored in tokens and converted to token_ids in order to be used as the inputs.'''

'''The below portion of code then loops through the sequences (both sequences at the same time using batch processing) and “primes” the model with our partial sentences. The variable output will be size batch_size by vocab_size. Remember this output is not a probability. After passing it through the softmax function, we can interpret it as a probability and sample from it.'''

temperature = 0.5 #1.0 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature
    probs = torch.exp(output)
    probs[:,0] = 0.0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1))
    x = torch.multinomial(probs,1)
    review.append(x.cpu().data.numpy()[:,0])

    ## predict the next word
    embed = model.embedding(x)

    h = model.lstm1(embed[:,0,:])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    output = model.decoder(h)

'''
Here is where we will actually generate the fake reviews. We use the previous output, perform the softmax function (assign probability of 0.0 to token id 0 to ignore the unknown token), randomly sample based on probs, save these indices to the list review, and finally get another output.'''

review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    print('\n')
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)
	
'''Here we simply convert the token ids to their corresponding string. Remember all of the indices need -1 to account for the unknown token we added before using it with imdb_dictionary.'''


'''

# temperature 1.0
a hugely influential , very strong , nuanced stand up comedy part all too much . this is a film that keeps you laughing and cheering for your own reason to watch it . the same has to be done with actors , which is surely `` the best movie '' in recent history because at the time of the vietnam war , ca n't know who to argue they claim they have no choice ... out of human way or out of touch with personal history . even during the idea of technology they are not just up to you . there is a balance between the central characters and even the environment and all of that . the book is beautifully balanced , which is n't since the book . perhaps the ending should have had all the weaknesses of a great book but the essential flaw of the 

i found it fascinating and never being even lower . spent nothing on the spanish 's particularly good filming style style . as is the songs , there 's actually a line the film moves so on ; a sequence and a couple that begins almost the same exact same so early style of lot of time , so the indians theme has not been added to the on screen . well was , the movie has to be the worst by the cast . i did however say - the overall feel of the film was superb , and for those that just do n't understand it , the movie deserves very little to go and lets you see how it takes 3 minutes to chilling . i must admit the acting was adequate , but `` jean reno '' was a pretty good job , he was very subtle 

***
Although these reviews as a whole don’t make a lot of sense, it’s definitely readable and the short phrases seem quite realistic. The temperature parameter from before essentially adjusts the confidence of the model. Using temperature=1.0 is the same as the regular softmax function which produced the reviews above. As the temperature increases, all of the words will approach having the same probability. As the temperature decreases, the most likely word will approach a probability of 1.0.
***

# temperature 1.3
a crude web backdrop from page meets another eastern story ( written by an author bought ) when it was banned months , i am sure i truly are curiosity ; i have always been the lone clumsy queen of the 1950 's shoved director richard `` an expert on target '' . good taste . not anything report with star 70 's goods , having it worked just equally attractive seem to do a moving train . memorable and honest in the case of `` ross , '' you find it fantasy crawford is strong literature , job suffering to his a grotesque silent empire , for navy turns to brooklyn and castle of obsession that has already been brought back as welles ' anthony reaches power . it 's totally clearly staged , a cynical sit through a change unconscious as released beer training in 1944 with mickey jones 

i wanted to walk out on featuring this expecting even glued and turd make . he genius on dialog , also looking good a better opportunity . anyway , the scene in the ring where christopher wallace , said things giving the least # 4 time anna hang earlier too leaves rick the blond doc from , walter from leon . is ironic until night with rob roy , he must 've been a mother . which are images striking to children while i think maybe this is not mine . but not in just boring bull weather sake , which set this by saying an excellent episode about an evil conspiracy monster . minor character with emphasis for blood deep back and forth between hip hop , baseball . as many red light figure hate americans like today 's life exercise around the british variety kids . nothing was added

***
Note here with a higher temperature, there is still some sense of structure but the phrases are very short and anything longer than a few words doesn’t begin to make much sense. Choosing an even larger temperature would result in random words being chosen from the dictionary.
***

## temperature 0.25
a little slow and i found myself laughing out loud at the end . the story is about a young girl who is trying to find a way to get her to go to the house and meets a young girl who is also a very good actress . she is a great actress and is very good . she is a great actress and she is awesome in this movie . she is a great actress and i think she is a great actress . i think she is a great actress and i hope she will get more recognition . i think she has a great voice . i think she is a great actress . i think she is one of the best actresses in the world . she is a great actress and i think she is a great actress . she is a great actress and

i was a fan of the original , but i was very disappointed . the plot was very weak , the acting was terrible , the plot was weak , the acting was terrible , the story was weak , the acting was horrible , the story was weak , the acting was bad , the plot was bad , the story was worse , the acting was bad , the plot was stupid , the acting was bad , the plot was stupid and the acting was horrible . i do n't know how anyone could have made this movie a 1 . i hope that it will be released on dvd . i hope it gets released on dvd soon . i hope that it will be released on dvd . i hope it gets released soon because it is so bad it 's good . i hope that

***
With a lower temperature, the predictions can get stuck in loops. However, it is interesting to note how the top review here happened to be a “good” review while the bottom review happened to be a “bad” review. It seems that once the model becomes confident with the tone of the review, it sticks with it. Remember that this language model was trained to simply predict the next word in 12500 positive reviews, 12500 negative reviews, and 50000 neutral reviews. It seems to naturally be taking into consideration the sentiment without explicitly being told to do so.
***

# temperature = 0.5
a very , very good movie , but it is a good movie to watch . the plot is great , the acting is good , the story is great , the acting is good , the story is good , there is plenty of action . i can not recommend this movie to anyone . i would recommend it to anyone who enjoys a good movie . i 'm not sure that the movie is not a good thing , but it is a good movie . it is a great movie , and a must see for anyone who has n't seen it . the music is great , the songs are great , and the music is fantastic . i do n't think i have ever seen a movie that is so good as the story line . i would recommend this movie to anyone who wants to 

i usually like this movie , but this is one of those movies that i could n't believe . i was so excited that the movie was over , i was very disappointed . it was n't that bad , but it was n't . it was n't even funny . i really did n't care about the characters , and the characters were so bad that i could n't stop laughing . i was hoping for a good laugh , but i was n't expecting much . the acting was terrible , the acting was poor , the story moved , and the ending was so bad it was just so bad . the only thing that kept me from watching this movie was the fact that it was so bad . i 'm not sure if it was a good movie , but it was n't . i was 

***
We’re not necessarily generating fake reviews here for any particular purpose (although there are applications that call for this). This is more to simply show what the model learned. It’s simple enough to see the accuracy increasing and the loss function decreasing throughout part 3a, but this helps us get a much better intuitive understanding of what the model is focusing on.
***

