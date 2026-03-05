# Salt-spreading problem
### Semester project for the biocomputing course at FIT VUT

Salt-spreading was defined in the scope of ROAR-NET (see [original repository](https://github.com/roar-net/problem-statements/tree/training-school-problems/problems/salt-spreading)).
The goal is to create an evolutionary algorithm able to solve arc routing problem with the specific set of constraints. The choice of algorithm is arbitrary, but the standard for
t#his type of problems is memetic algorithm, which combines traditional genetic programming with some kind of optimization (e.g. local search methods) applied after each step.

Since the algorighm is stochastic, the solution should include conducting experiments and their subsequent statistical processing. This may involve the usage of different 
genetic operators, adjusting params etc. The repository provides four datasets, two of which are smaller and intended for initial testing of the solution. The other two are based on real data and do not need to be solved,
although it is recommended to take the time to adjust the parameters for more effective performance. It is alo required to compare two different approaches for visiting depot:
1. Necessary visits are inserted at the stage of interpretation of the result.
2. Visiting depot is percieved as a special operation, which makes it the part of the omptimization process. 

The first challenge is to define the method of the chromosome encoding. One proposed is to encode only salting actions, ignoring other movement along a graph. This would also require
the preprocessing of nodes: we should know how long it takes to reach every node from the current state. 


### References

[Solving Various Classes of Arc Routing Problems with a Memetic Algorithm-based Framework](https://optimization-online.org/wp-content/uploads/2023/11/Arc_Routing_MA.pdf)

[Memetic algorithm with non-smooth penalty for capacitated arc routing problem](https://www-sciencedirect-com.ezproxy.lib.vutbr.cz/science/article/pii/S0950705121002203)
