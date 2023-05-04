# MAP AI
###  super simplified MAP environment with AI model
###### 

## Emphasis
AI is the simulation of human intelligence processes by machines, especially computer systems.  Recently, the term Artificial Intelligent (AI) has become more common even outside of the technology industry.   A couple months ago, there was a headline in the Japan Economic News Paper, saying that AI can find the quality of taste in milk made from cabbage.  AI technology is now used in many areas.   

 AI  helps us solve problems of various complexities.  As a final project for CS4300, I choose to create a very simplified version of Map AI using A star search algorithm in python.  Here in St George, I have to drive for a living though driving is not something I am good at.  I always use GPS on google map to find the fastest path to the destination, since I am still not very familiar with the directions or the rush hours.  Through this project, I would like to create AI to find the shortest path from where you are to the destination. 

## Simplified Map
* Generates map with randomly placed obstacles 
* Set the size of the map (square - width and * height are the same length)
* Goal is always at [-1, -1]
* Start is always at [0, 0]

## The AI Algorithms

* MinMax
* AlphaBeta 
* Random 

### The Percepts
* Map
* Start location
* Final destination

### The Rewards
* every step: -1
* when Mario die: -1000

### The Actions
Directions
1. North
2. South
3. West
4. East
5. North-west
6. North-east
7. South-west
8. South-east


## Links

https://github.com/ksawada1/ksawada-school-projects/tree/main/MapAI