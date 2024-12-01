from time import time
import time, enum, math, random
import numpy as np

from numpy.random import randint

import pandas as pd
import subprocess


def run_cpp_simuation(n_threads, pop, neighbors, timesteps, initially_influential_prop, initially_infected_prop,
                      initially_vaccinated_prop, p_samples_reshare, p_samples_reject, p_samples_online,
                      weight_chrono, weight_belief, weight_pop, weight_random, filename):

    cpp_file_path = 'main.cpp'
    exe_file_path = 'main'

    # Compile the C++ code
    compile_command = ['g++', cpp_file_path, '-o', exe_file_path]
    # for server
    # compile_command = ['g++', cpp_file_path, '-o', exe_file_path, "-lpthread"]
    subprocess.run(compile_command, check=True)

    # Call the compiled C++ program and pass the arguments as input
    cpp_command = ['./' + exe_file_path]
    process = subprocess.Popen(cpp_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    input_data = (str(n_threads) + '\n' + str(pop) + '\n' + str(neighbors) + '\n' + str(timesteps) + '\n' +
                  str(initially_influential_prop) + '\n' + str(initially_infected_prop) + '\n' +
                  str(initially_vaccinated_prop) + '\n' + str(p_samples_reshare) + '\n' +
                  str(p_samples_reject) + '\n' + str(p_samples_online) + '\n' + str(weight_chrono) + '\n' +
                  str(weight_belief) + '\n' + str(weight_pop) + '\n' + str(weight_random) + '\n' + str(filename))

    output, _ = process.communicate(input=input_data.encode())

    # Clean up the compiled executable file
    subprocess.run(['rm', exe_file_path])

    return float(output.strip())


def run_abm(preject=0.01, preshare=0.01, ponline=0.1, weight_chrono=0.25,
            weight_belief=0.25, weight_pop=0.25, weight_random=0.25,
            init_inf=0.1, init_vacc=0.1, init_infl=0.1,
            steps=100, n=3, df_hours=pd.DataFrame, filename="filtered_df.csv", n_threads=1):
    """
    calls n simulations of the network model and outputs the average
    distance between the simulation output and the real world data
    """

    for n in range(n):

        start = time.time()
        distance = run_cpp_simuation(n_threads, population, neighbours, steps, init_infl, init_inf, init_vacc, preshare,
                                             preject, ponline, weight_chrono, weight_belief, weight_pop, weight_random, filename)
        distances.append(distance)
        end = time.time()

    avg_distance = np.mean(distances)

    return avg_distance


def get_samples(n_samples, params_range, dp=4):
    """ get samples for calibration of the probabilities """

    samples = []
    for i in params_range:
        vals = list(np.linspace(i[0], i[1], n_samples))
        vals = [round(i, dp) for i in vals]
        samples.append(vals)


    total_samples = []
    for i in samples[0]:
        for j in samples[1]:
            candidate = [i, j]
            total_samples.append(candidate)

    total_samples = sorted(total_samples)
    total_samples = [total_samples[i] for i in range(len(total_samples)) if
                     i == 0 or total_samples[i] != total_samples[i - 1]]

    return total_samples


def get_weights(n_samples, params_range, n_dims=4, dp=4):
    """ get samples for optimisation of weights """

    samples = []
    for i in params_range:
        vals = list(np.linspace(i[0], i[1], n_samples))
        vals = [round(i, dp) for i in vals]
        samples.append(vals)


    total_samples = []
    for i in samples[0]:
        for j in samples[1]:
            for k in samples[2]:
                for l in samples[3]:
                    sum = i + j + k + l
                    if sum == 1:
                        candidate = [i, j, k, l]
                        total_samples.append(candidate)

    total_samples = sorted(total_samples)
    total_samples = [total_samples[i] for i in range(len(total_samples)) if
                     i == 0 or total_samples[i] != total_samples[i - 1]]

    return total_samples


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = np.random.dirichlet(np.ones(n_bits), size=n_pop).tolist()
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        start_gen = time.time()
        print(f"Generation: {gen}")
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
            end_gen = time.time()
            # print(f"time for gen: {end_gen - start_gen} s")
        # replace population
        pop = children
    return [best, best_eval]


# objective function or fitness function
def objective(candidate):
    print(f"Candidate to Evaluate: {candidate}")
    y_dict = {}
    euclidean_dist_list = []
    for story in stories:
        y_evaluated = np.array([])
        xx = probability_dict[story]
        df_hours = data[data.tweet_id == story].reset_index()
        init_inf = df_hours[df_hours.steps == 1].prop_tweets_inf_rw[1]
        init_vacc = init_inf
        init_infl = 0.1
        steps = df_hours.steps.nunique() - 1
        y_evaluated = np.append(y_evaluated, run_abm(preject=xx[0],
                                                     preshare=xx[1],
                                                     weight_chrono=candidate[0],
                                                     weight_belief=candidate[1],
                                                     weight_pop=candidate[2],
                                                     weight_random=candidate[3],
                                                     init_inf=init_inf,
                                                     init_vacc=init_vacc,
                                                     init_infl=init_infl,
                                                     ponline=0.2,
                                                     steps=steps,
                                                     n=n_sims,
                                                     df_hours=df_hours))

        euclidean_dist_list.append(y_evaluated)

    avg_dist = np.average(euclidean_dist_list)
    print(f"Average Distance: {avg_dist}")

    return avg_dist


# Main
if __name__ == '__main__':
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

    data = pd.read_csv('[INPUT DATA]', dtype={'tweet_id': str}).dropna()

    print("Data Summary")

    print(f"No. Case-Studies: {data.tweet_id.nunique()}")

    print(f"Case-Studies: {data.tweet_id.unique()}")

    stories = data.tweet_id.unique()

    # Calibration of Probabilities

    params_range = np.array([preject_list, preshare_list, ponline_list, psigma_list])

    # Generate y values - Weights

    np.random.seed(1234)

    weights_range = np.array([weights_chrono, weights_belief, weights_pop, weights_random])
    print(f"Weights range: {weights_range}")

    # Total sample space
    total_weights = get_weights(10, weights_range)

    # Generate x values - Probabilities

    np.random.seed(1234)
    params_range = np.array([preject_list, preshare_list])
    print(f"Probability range: {params_range}")

    # Total sample space
    x_evaluated = get_samples(10, params_range)  # Total sample space

    y_dict = {}

    for story in stories:
        print(f"Story: {story}")

        start_story = time.time()
        y_evaluated = np.array([])

        df_hours = data[data.tweet_id == story].reset_index()
        filename = "filtered_df.csv"
        df_hours.to_csv(filename)
        print(df_hours.head())

        init_inf = df_hours[df_hours.steps == 1].prop_tweets_inf_rw[1]
        init_vacc = init_inf
        init_infl = 0.1
        steps = df_hours.steps.nunique() - 1
        for i, xx in enumerate(x_evaluated):

            start_y = time.time()
            y_evaluated = np.append(y_evaluated, run_abm(preject=xx[0], preshare=xx[1], ponline=0.2,
                                                         init_inf=init_inf, init_vacc=init_vacc, init_infl=init_infl,
                                                         df_hours=df_hours, steps=steps, n=n_sims))
            end_y = time.time()

        y_dict[story] = y_evaluated
        end_story = time.time()



    probability_dict = dict()

    for story in stories:
        y_eval = y_dict[story]
        y_opt = min(y_eval)
        loc_y_opt = np.where(y_eval == y_opt)[0][0]
        opt_params = x_evaluated[loc_y_opt]
        probability_dict[story] = opt_params

        print(f"For Story: {story} Opt: {opt_params} where y: {y_opt}")

    print(probability_dict)

    # Optimisation of Weights (GA)
    start_total = time.time()

    n_sims = 1

    # population size
    n_pop = 10

    # length candidate
    n_bits = 4

    # no generations
    n_iter = 5

    # crossover rate
    r_cross = 0.9

    # mutation rate
    r_mut = 0.1

    result = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)

    print(f"Result: {result}")

    end_total = time.time()

    print(f"GA Runtime: {end_total - start_total} s")
