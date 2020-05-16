**Distributional Deep Q-Learning Project**

Distributional RL is an interesting recent development of value based RL, where the algorithm models
the entire distribution of Q values, instead of only the expectation as in the usual Bellman equation. It
seems to lead to great empirical improvements. 

**Objective:**
*Implement the paper (the C51 algorithm) in pytorch and test it with Atari
game Pong and show reasonable score of average >= 15*


**Implementation Details**

We first use a deep neural network to represent the value distribution. Since the inputs are screen pixels, the first
4 layers are convolutional layers . The neural network outputs values of distribution predictions for each action.
Each set of prediction is a softmax layer with 51 units. In our case of Pong environment atoms is the number
of discrete values = 51. First we sample a minibatch of sample paths from the Experience Replay buffer and
initialize the corresponding states, reward, and targets variables. Then we store the probability mass of the value
distribution and carry out a forward pass to get the next state distributions. The model outputs distributions for
each action. Our target is to get the one with the largest expected value to carry out the next update. We then
compute the target distribution by scaling with discount factor shifting by reward value. Then we project it to the 51
discrete supports. We then perform minimize the mean square error loss using Adam optimizer. At the beginning
of training, the agent performs only random actions and as we can see from the plot, the agent gets a reward of
around -20 as it is losing most of the time. After around 1000 timesteps the agent already learns to hit the ball and
is able to score its first points. We found that the initial learning curve was increased monotonically around 400000
time steps, but learning rate was slow seems to be because of the exploration policy. After that as the agent starts
gathering experience and reducing the exploration strategy over time we saw a rate improvement till 500000 time
steps. After that the performance saturates to score of around 18.

**Result:**
*Achieved score of around 18 for Atari pong.* 