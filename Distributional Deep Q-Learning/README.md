*Distributional Deep Q-Learning*

Distributional RL is an interesting recent development of value based RL, where the algorithm models
the entire distribution of Q values, instead of only the expectation as in the usual Bellman equation. It
seems to lead to great empirical improvements. This is an extension to the deep RL assignment.

Objective:
***To implement the paper (the C51 algorithm) in pytorch and test it with simple Atari
games, e.g. Pong, Breakout, etc., and show reasonable scores (average >= 15 for Pong, >= 100 for
Breakout).***

Got reasonable success in Pong by achieving score around 18 for pong. While Pong scores met standard scores, there was no luck in sight with breakout even after trying a multitude of hyperparameters. 
