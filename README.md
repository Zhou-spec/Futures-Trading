## Futures Trading 
This repository is part of the internship project at Galaxy Derivatives Capital Management Co Ltd. The project aims to develop an RL framework for futures trading using machine learning techniques and financial market data. However, this repository will only contain publicly available information
### Baseline Model
The baseline model is adapted from [Deep Reinforcement Learning for Active High Frequency Trading](https://arxiv.org/pdf/2101.07107.pdf) The following list summarizes the characteristics of the baseline RL model
1. The agent can hold or short at most one unit of the asset (in this case, one lot of futures)
2. State: Each state consists of 21 variables, which include the first 5 bid and ask prices along with their corresponding volumes. Additionally, the state incorporates the position of the agent, (-1 is short position, 0 is no position, and 1 is long position). <br>
  (bid price 1-5, bid volume 1-5, ask price 1-5, ask volume 1-5, position)
3. Action: 

### Useful Materials
The followings are some materials that I found useful. Some may be blogs written in Chinese but one can easily find an alternative in English. 
1. [Explanation of TRPO (Chinese)](https://www.zhihu.com/question/366605427/answer/1048153125)
2. [Deepen your Understanding of PPO (Chinese)](https://zhuanlan.zhihu.com/p/614115887)
