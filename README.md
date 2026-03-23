![alpine_877](https://user-images.githubusercontent.com/4012178/116814864-1b1a1580-ab5b-11eb-97e6-1441af4ececa.png)

# ch.alpine.subare

Library for reinforcement learning in Java 17.

![](https://github.com/datahaki/subare/actions/workflows/mvn_test.yml/badge.svg)

Repository includes algorithms, examples, and exercises from the 2nd edition of [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton, and Andrew G. Barto.

Our implementation is inspired by the
[python code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
by Shangtong Zhang, but differs from the reference in two aspects:

* the algorithms are implemented **separate** from the problem scenarios
* the math is in **exact** precision which reproduces symmetries in the results in case the problem features symmetries

## Algorithms

* Iterative Policy Evaluation (parallel, in 4.1, p.59)
* *Value Iteration* to determine V*(s) (parallel, in 4.4, p.65)
* *Action-Value Iteration* to determine Q*(s,a) (parallel)
* First Visit Policy Evaluation (in 5.1, p.74)
* Monte Carlo Exploring Starts (in 5.3, p.79)
* Contant-alpha Monte Carlo
* Tabular Temporal Difference (in 6.1, p.96)
* *Sarsa*: An on-policy TD control algorithm (in 6.4, p.104)
* *Q-learning*: An off-policy TD control algorithm (in 6.5, p.105)
* Expected Sarsa (in 6.6, p.107)
* Double Sarsa, Double Expected Sarsa, Double Q-Learning (in 6.7, p.109)
* n-step Temporal Difference for estimating V(s) (in 7.1, p.115)
* n-step Sarsa, n-step Expected Sarsa, n-step Q-Learning (in 7.2, p.118)
* Random-sample one-step tabular Q-planning (parallel, in 8.1, p.131)
* Tabular Dyna-Q (in 8.2, p.133)
* Prioritized Sweeping (in 8.4, p.137)
* Semi-gradient Tabular Temporal Difference (in 9.3, p.164)
* True Online Sarsa (in 12.8, p.309)

## 👥 Contributors

Jan Hakenberg, Christian Fluri

## Publications

* [*Learning to Operate a Fleet of Cars*](https://www.research-collection.ethz.ch/handle/20.500.11850/304517)
by Christian Fluri, Claudio Ruch, Julian Zilly, Jan Hakenberg, and Emilio Frazzoli

## References

* [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
by Richard S. Sutton, and Andrew G. Barto
