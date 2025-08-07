
## Welcome to moco.

*Note: this repository is under development.*

moco is purpose-built for performance-minded engineers.

The optimization implemented in this library is attach simple rules to the model.
At run time, for a given rule, for a given data point, check whether the data point satisfies 
the rule. If it does, output the class assigned to that rule, and if it does not,
evaluate the model as usual. 

This results in very clear logic whether the rule makes the average latency increase 
or decrease.

For example, if a model takes 0.001 second each for a dataset of 1M data points, and for 20% of the data points, I can actually compute a prediction in 0.0005 seconds, then this will go from taking 1000 seconds to taking 800000 * 0.001 + 200000 * 0.0005 = 900 seconds. 

This assumes that the time the rule takes to decide whether a data point is easy is negligible compared to the time it takes to evaluate the model. Thus, the complexity of the rule matters.

Finally, it's important that we maintain accuracy. These rules when activated do not result in different predictions than the original model.

#### Note that `get_fast_rules` does call an API automatically. That API is not currently deployed rendering this software unusable.
*If data privacy is a concern, please do not use this software as is, but rather file an issue or contact me.*


See the examples folder for usage.
The project page: https://compressmodels.github.io
