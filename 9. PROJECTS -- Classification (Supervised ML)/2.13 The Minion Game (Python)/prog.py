'''

Kevin and Stuart want to play the 'The Minion Game'.

Game Rules

Both players are given the same string, .
Both players have to make substrings using the letters of the string .
Stuart has to make words starting with consonants.
Kevin has to make words starting with vowels.
The game ends when both players have made all possible substrings.

Scoring
A player gets +1 point for each occurrence of the substring in the string .

For Example:
String  = BANANA
Kevin's vowel beginning word = ANA
Here, ANA occurs twice in BANANA. Hence, Kevin will get 2 Points.

Your task is to determine the winner of the game and their score.

'''

def con_start_function(data):
    con_start=-1
    for i in range(len(data)):
        if data[i][1]==0:
            con_start=i
            break
    return con_start

def vow_start_function(data):
    vow_start=-1
    for i in range(len(data)):
        if data[i][1]==1:
            vow_start=i
            break
    return vow_start

def minion_game(string):
    
    input_list = [x for x in string]
    vowels = ['A', 'E', 'I', 'O', 'U']
    input_check = []

    for i in input_list:
        if i not in vowels:
            input_check.append([i,0])
        else:
            input_check.append([i,1])
    
    total_length = len(input_check)
    con_score=0
    current_data = input_check.copy()
    count=0
    while True:
        con_start = con_start_function(current_data)
        if con_start == -1:
            break
        current_data = current_data[con_start:]
        con_score += len(current_data)
        count += 1    
        if count>total_length:
            break
        current_data = current_data[1:]   


    vow_score=0
    current_data = input_check.copy()
    count=0
    while True:
        vow_start = vow_start_function(current_data)
        if vow_start == -1:
            break
        current_data = current_data[vow_start:]
        vow_score += len(current_data)
        count += 1    
        if count>total_length:
            break
        current_data = current_data[1:]   
        
        
    if con_score == vow_score:
        print('Draw')
    elif con_score > vow_score:
        print('Stuart', con_score)   
    elif con_score < vow_score:
        print('Kevin', vow_score)   


if __name__ == '__main__':
    s = input()
    minion_game(s)