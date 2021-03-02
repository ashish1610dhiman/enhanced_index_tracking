#Work Record

### Nov 22, 2020
* Modified Linear Relaxation script to accommodate iterative experimentation
* Created notebook for experimentation 
* _Objective printed while optimisation, is devoid of constant_


### Nov 23
**For comparing performance b/w models, we can give index as combination of some weights, therefore confirming that index can be tracked, and then judge model's performance against this**
* Creating class to test experiments

### Nov 26
* Added DVC for version control of experiment data
* **Experiment 2**: Try to replicate table for paper

#### Idea for testing the 3 approaches (x 2 for reduced/non-reduced data)


For the 3 approaches,
1. Non-Linear (requires benchmark weights)
2. EIT Basic (does not require benchmark weights)
3. EIT improved (does not require benchmark weights)

On our index tracking data, we shall create 1 artificial benchmark index, using approach similar to section-1
We track this index using 3 approaches, divide Out of Tracking into 3/4 parts

Ideally comparison should be between the best of 3 models, Best as defined by family of parameters. 

### Dec 8
The Dimension reduction is tested only for mean-variance model, how does it apply to Linear (or L_1 risk)

### Dec 9
Formulate combined approach and add results


### Feb 28
Definition of excess return and Tracking Error is slightly different between EIT_basic and EIT_dual.
But performance metric is same
