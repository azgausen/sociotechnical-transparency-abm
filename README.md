# sociotechnical-transparency-abm
This repository contains the core logic for the ABM used in Gausen, A., Guo, C. & Luk, W. An approach to sociotechnical transparency of social media algorithms using agent-based modelling. AI Ethics (2024). https://doi.org/10.1007/s43681-024-00527-1. 

Note: the published results where run on a version of the code but the core logic should be the same.

**Summary:**

Main.cpp: 
contains the C++ code with the ABM

Run_abm.py: 
contains the python code with the parameter inputs, calibration, genetic algorithm and wrapper to run the C++ code.

**Parameters:**

    population = 250
    neighbours = 25
    n_sims = 1
    n_posts = 10

    preject_list = [0.01, 0.05]
    preshare_list = [0.01, 0.05]
    ponline_list = [0.1, 0.1]
    psigma_list = [0.001, 0.05]

    weights_chrono = [0.1, 1.0]
    weights_belief = [0.1, 1.0]
    weights_pop = [0.1, 1.0]
    weights_random = [0.1, 1.0]
    n_threads = 8
    init_infl = 0.1

    n_sims = 1
    n_pop = 10
    n_bits = 4
    n_iter = 5
    r_cross = 0.9
    r_mut = 0.1


**Inputs and Outputs:**
Input: Need a CSV with real world or dummy data on multiple tweets.

Output: Runs the optimum probabilities for each case study, optimum weights for all case studies, distance between simulated and real data.

**How to Run:**

1. Clone repo

2. Navigate to cloned repository

3. Make sure you have dependencies (python, g++, etc)

4. In run_abm.py update [INPUT DATA] with the name of your csv.
   
5. Run command: python3 run_abm.py


**Data:**

Column headings must be in alphabetical order.
Column Headings: [counts	prop_tweets_inf_rw	steps	tweet_id]


**Licensing and Attribution:**

This repo is under a  BSD 3-Clause license. For proper attribution when using this code in any publications or research outputs, please cite our paper:

    @article{gausen2024approach,
      title={An approach to sociotechnical transparency of social media algorithms using agent-based modelling},
      author={Gausen, Anna and Guo, Ce and Luk, Wayne},
      journal={AI and Ethics},
      pages={1--19},
      year={2024},
      publisher={Springer}
    }

Suggested In-text Citation: Gausen, A., Guo, C. & Luk, W. An approach to sociotechnical transparency of social media algorithms using agent-based modelling. AI Ethics (2024). https://doi.org/10.1007/s43681-024-00527-1



