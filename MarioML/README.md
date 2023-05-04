# Mario ML
###  A super basic version of super mario environment with ML model
###### 

## Emphasis
Super Mario Land is a platform game created by Nintendo and it tells a story about Mario, a brave plumber who lives in the land of the Mushroom Kingdom and he has the role of saving the princess from the hands of the villain Browser.

In this game, there are stages by levels, and some stages are hard to make it to the goal.  

This environment only has a stage. The model can understand the environment and take the action it thinks best to reach the goal.  The model recognizes the gummas who are Mario’s enemy in the stage.  Unlike humans, the model doesn’t get emotionally upset and can make rational/right decisions.  Also the model can observe and pay attention to everything in the stage(environment) for right actions.  With ML, every step Mario takes, he gets smarter and gets closer to the goal by getting rewards and penelty.


## The ML Methods

* Q-Function (Reinforcement Learning) 

### The Percepts
* Mario (life : 1)
* 2 Gummas

### The Rewards
* every step: -1
* when Mario die: -1000

### The Actions

1. go forwards
2. go backwards
3. jump
4. wait


## Links

https://github.com/ksawada1/ksawada-school-projects/tree/main/MarioML