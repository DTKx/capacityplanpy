<!-- Project -->
# CapacityPlanPy
Project developed during lecture of Genetic Algorithms that consists on developing resilient production schedules for four pharmaceutical products over three years, considering:
- Multiobjectives optimization:
    - Maximization of Production
    - Minimization of stock deficit
- Restricted to:
    - Median of backlogs equal to 0

The current project focus on the reproduction of the [Article](https://www.sciencedirect.com/science/article/abs/pii/S0098135418312146?via%3Dihub):

Karolis Jankauskas, Suzanne S. Farid,
Multi-objective biopharma capacity planning under uncertainty using a flexible genetic algorithm approach,
Computers & Chemical Engineering,
Volume 128,
2019,
Pages 35-52,
ISSN 0098-1354,
https://doi.org/10.1016/j.compchemeng.2019.05.023.
(https://www.sciencedirect.com/science/article/pii/S0098135418312146)
Abstract: This paper presents a flexible genetic algorithm optimisation approach for multi-objective biopharmaceutical planning problems under uncertainty. The optimisation approach combines a continuous-time heuristic model of a biopharmaceutical manufacturing process, a variable-length multi-objective genetic algorithm, and Graphics Processing Unit (GPU)-accelerated Monte Carlo simulation. The proposed approach accounts for constraints and features such as rolling product sequence-dependent changeovers, multiple intermediate demand due dates, product QC/QA release times, and pressure to meet uncertain product demand on time. An industrially-relevant case study is used to illustrate the functionality of the approach. The case study focused on optimisation of conflicting objectives, production throughput, and product inventory levels, for a multi-product biopharmaceutical facility over a 3-year period with uncertain product demand. The advantages of the multi-objective GA with the embedded Monte Carlo simulation were demonstrated by comparison with a deterministic GA tested with Monte Carlo simulation post-optimisation.
Keywords: Multi-objective; Uncertainty; Biopharmaceutical; Capacity planning; Scheduling; Genetic algorithm

<!-- Motivation -->
Product development is strategic to the long term development of the company, however it is a complex process especially given the interface with multiple company areas, therefore being tipically multiobjective.
In case of pharmaceutical products additional complexity arise given regulatory approval process, requiring  product capacity planning even during early development phases.
One of the main challenges in capacity planning is to increase resilience given demand fluctuations.

Therefore the cited article proposes an interesting approach to solve the problem using Genetic Algorithms coupled with Monte Carlo simulations of demand to generate solutions in pareto front and more resilient.

<!-- TODO: Installation -->
<!-- TODO: Tests -->
<!-- TODO: How to use? -->
<!-- Credits -->

[Article](https://www.sciencedirect.com/science/article/abs/pii/S0098135418312146?via%3Dihub):

Karolis Jankauskas, Suzanne S. Farid,
Multi-objective biopharma capacity planning under uncertainty using a flexible genetic algorithm approach,
Computers & Chemical Engineering,
Volume 128,
2019,
Pages 35-52,
ISSN 0098-1354,
https://doi.org/10.1016/j.compchemeng.2019.05.023.
(https://www.sciencedirect.com/science/article/pii/S0098135418312146)
Abstract: This paper presents a flexible genetic algorithm optimisation approach for multi-objective biopharmaceutical planning problems under uncertainty. The optimisation approach combines a continuous-time heuristic model of a biopharmaceutical manufacturing process, a variable-length multi-objective genetic algorithm, and Graphics Processing Unit (GPU)-accelerated Monte Carlo simulation. The proposed approach accounts for constraints and features such as rolling product sequence-dependent changeovers, multiple intermediate demand due dates, product QC/QA release times, and pressure to meet uncertain product demand on time. An industrially-relevant case study is used to illustrate the functionality of the approach. The case study focused on optimisation of conflicting objectives, production throughput, and product inventory levels, for a multi-product biopharmaceutical facility over a 3-year period with uncertain product demand. The advantages of the multi-objective GA with the embedded Monte Carlo simulation were demonstrated by comparison with a deterministic GA tested with Monte Carlo simulation post-optimisation.
Keywords: Multi-objective; Uncertainty; Biopharmaceutical; Capacity planning; Scheduling; Genetic algorithm

<!-- TODO: License -->
