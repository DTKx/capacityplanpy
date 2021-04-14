<!-- Project -->
# CapacityPlanPy
Project developed during lecture of Genetic Algorithms consisting on developing resilient production schedules for four pharmaceutical products over three years, considering:
- Multiobjectives optimization:
    - Maximization of Production
    - Minimization of stock deficit
- Restricted to:
    - Median of backlogs equal to 0

The optimized solutions of the pareto front can be visualized below:
![Example of a pareto front found.](/images/pareto_front_example.png)

Moreover an example of a planification can be seen below:
![Example of a planification, with the number of batches to be produced.](/images/planningX.png)

The current project focus on the reproduction of the [Article](https://www.sciencedirect.com/science/article/abs/pii/S0098135418312146?via%3Dihub):
```
@article{JANKAUSKAS201935,
title = {Multi-objective biopharma capacity planning under uncertainty using a flexible genetic algorithm approach},
journal = {Computers & Chemical Engineering},
volume = {128},
pages = {35-52},
year = {2019},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2019.05.023},
url = {https://www.sciencedirect.com/science/article/pii/S0098135418312146},
author = {Karolis Jankauskas and Suzanne S. Farid},
keywords = {Multi-objective, Uncertainty, Biopharmaceutical, Capacity planning, Scheduling, Genetic algorithm},
abstract = {This paper presents a flexible genetic algorithm optimisation approach for multi-objective biopharmaceutical planning problems under uncertainty. The optimisation approach combines a continuous-time heuristic model of a biopharmaceutical manufacturing process, a variable-length multi-objective genetic algorithm, and Graphics Processing Unit (GPU)-accelerated Monte Carlo simulation. The proposed approach accounts for constraints and features such as rolling product sequence-dependent changeovers, multiple intermediate demand due dates, product QC/QA release times, and pressure to meet uncertain product demand on time. An industrially-relevant case study is used to illustrate the functionality of the approach. The case study focused on optimisation of conflicting objectives, production throughput, and product inventory levels, for a multi-product biopharmaceutical facility over a 3-year period with uncertain product demand. The advantages of the multi-objective GA with the embedded Monte Carlo simulation were demonstrated by comparison with a deterministic GA tested with Monte Carlo simulation post-optimisation.}
}
```
Obs. Project Underdevelopment.
<!-- Motivation -->
# Motivation
## Product Development Multiobjectives
Product development is strategic to the long term development of the company, however it is a complex process especially given the interface with multiple company areas, as a result the combination might not lead to the optimal for the company. Therefore being tipically multiobjective. 

## Pharmaceutical Product Development
In case of pharmaceutical products additional complexity arise especially given regulatory approval process, requiring  product capacity planning even during early development phases.
One of the main challenges in capacity planning is to increase resilience given demand fluctuations.

## Resilient Product Capacity Planning through Genetic Algorithms
Given the context, more resilient product capacity planning strategies might optmize assets and resources. For instance: reducing financial risks, production costs, inventory costs, costs given unexpected stops/maintenance and backlog costs.

In light of this, Genetic Algorithms (GAs) proposes an interesting alternative to traditional optimization methods. Given that it may:
- Allow modelling of complex problems (e.g. non continuous, multiple local minima, maximum), which might not have a formal mathematical formula.
- Not require large amounts of data
- Proposes an alternative to exiting Local Minima/Maximum
- Proposes a more resilient planning solutions given utilization of Monte Carlo Simulations of demand

Therefore the cited article proposes an interesting approach to solve the problem using Genetic Algorithms coupled with Monte Carlo simulations of demand to generate more resilient solutions in pareto front considering multiobjectives.

### Intuition of Genetic Algorithms
GAs are heuristics based algorithms tipically used for optmization of NP hard problems and is bioinspired on the concept of evolving groups of solutions (populations) as method to select the most suitable solutions.
As in human kind evolutionary theory, human populations evolved from monkeys by series of genetic operations such as crossover and mutation and were selected by survival of individuals with most suitable characteristics. Those selected individuals with different characteristics would continue to evolve on next generations through their offsprings towards a more prone to survival population.

GAs share the same principles: 
- Starts with an initial population of solutions to the complex problem 
- Evolves the population using operators e.g. mutation and crossover, leveraging from diversity of solutions.
- Finally selects suitable solutions to continue further evolving the populations of solutions.

<!-- TODO: Installation -->
<!-- TODO: Tests -->
<!-- TODO: How to use? -->
<!-- Credits -->
# Credits
[Article](https://www.sciencedirect.com/science/article/abs/pii/S0098135418312146?via%3Dihub):
```
@article{JANKAUSKAS201935,
title = {Multi-objective biopharma capacity planning under uncertainty using a flexible genetic algorithm approach},
journal = {Computers & Chemical Engineering},
volume = {128},
pages = {35-52},
year = {2019},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2019.05.023},
url = {https://www.sciencedirect.com/science/article/pii/S0098135418312146},
author = {Karolis Jankauskas and Suzanne S. Farid},
keywords = {Multi-objective, Uncertainty, Biopharmaceutical, Capacity planning, Scheduling, Genetic algorithm},
abstract = {This paper presents a flexible genetic algorithm optimisation approach for multi-objective biopharmaceutical planning problems under uncertainty. The optimisation approach combines a continuous-time heuristic model of a biopharmaceutical manufacturing process, a variable-length multi-objective genetic algorithm, and Graphics Processing Unit (GPU)-accelerated Monte Carlo simulation. The proposed approach accounts for constraints and features such as rolling product sequence-dependent changeovers, multiple intermediate demand due dates, product QC/QA release times, and pressure to meet uncertain product demand on time. An industrially-relevant case study is used to illustrate the functionality of the approach. The case study focused on optimisation of conflicting objectives, production throughput, and product inventory levels, for a multi-product biopharmaceutical facility over a 3-year period with uncertain product demand. The advantages of the multi-objective GA with the embedded Monte Carlo simulation were demonstrated by comparison with a deterministic GA tested with Monte Carlo simulation post-optimisation.}
}
```
<!-- TODO: License -->
