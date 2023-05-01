Final Project - MAP/Path PEAS
Kumiko Sawada


Problem: 
My morning is very busy.  Even after I finally leave the house,  I have to drop my daughter off at school, then go to the class at UT.  However, I always wonder if there is an shortest path in the morning route without using google map gps.  In this project, I would like to create a map to find the shortest route from start point to the destination using AI.   	

	AI automatically evaluates the shortest path to the destination using A * search.

The Percepts (Sensors):
map, edge, start location, final destination, direction(up, up_right, up_left, right, left, down, down_right, down_left)

The Actions (Actuators):
Evaluate the shortest way to go if you go north, south, west, east
Then move forward to take the shortest path

The Percept:
The Environment is:
	Observability : Fully Observable 
	You can access to a whole map to the destination

	Uncertainly : Stochastic 
	The best path would not always be the same depending on road conditions, time of the day, traffic, and how many stops you make.

	Duration : Episodic
	Each map has different data and Agent re-calculate for the other map.

	Stability : Dynamic
	The map with the road environment and the path are always changing.

	Granularity : Discrete
	The map is not infinite.  Once you reach the goal, it is done.
	
	Participants : Single Agent 
	Only one agent can calculate the best time for the map using an A* search algorithm.
 
	Knowledge : Known.
	The formula is known.  Since it uses many kinds of data from everywhere, it is hard for humans to evaluate however it is possible using formulas.


The Performance Measure:
	Get the shortest path from start to the destination - 

