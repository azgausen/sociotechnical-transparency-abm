#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <utility>
#include <chrono>
#include <pthread.h>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

std::map<std::string, std::vector<float> > readCSV(const std::string &filename)
{
    std::map<std::string, std::vector<float> > data;
    std::ifstream file(filename);
    std::string line, key;
    float value;
    bool firstLine = true;

    while (getline(file, line))
    {
        std::istringstream s(line);
        if (firstLine)
        {
            while (getline(s, key, ','))
            {
                data[key] = std::vector<float>();
                if (s.peek() == ',')
                    s.ignore();
            }
            firstLine = false;
        }
        else
        {
            int col = 0;
            for (std::map<std::string, std::vector<float> >::iterator it = data.begin(); it != data.end(); ++it)
            {
                getline(s, key, ',');
                if (key.empty())
                    break;
                std::istringstream iss(key);
                float value;
                iss >> value;
                it->second.push_back(value);
                col++;
            }
        }
    }
    file.close();
    return data;
}

// Bayesian Belief Updating (removed from Agent Class)
double bayesian_belief_updating(double X, double p_h, double sigma_own, double mu_own) {
    double sigma_true = 0.5;
    double mu_true = 0.5;

    double P_H = p_h;
    double P_E = 1/(sigma_true * sqrt(2*M_PI)) * exp(pow((X-mu_true),2)/(2*pow(sigma_true,2)));

    double P_E_H = 0;
    if (sigma_own == 0) {
        P_E_H = mu_own;
    }
    else {
        P_E_H = 1/(sigma_own * sqrt(2*M_PI)) * exp(pow((X-mu_own),2)/(2*pow(sigma_own,2)));
    }

    double P_H_E = P_H * (P_E_H/P_E);

    return P_H_E;
}

// Get Curation Type

std::string get_curation_type(std::vector<double> weight_probs, std::vector<std::string> types) {
    // choose curation type based on weights (note: this implementation is very
    // simplified as it picks a curation type for each post)
    std::random_device rand;
    double randNumber = rand() % 1;
    double sum = 0;
    int result = 0;
    for (int i = 0; i < weight_probs.size(); i++) {
        sum += weight_probs[i];
        if (randNumber < sum) {
            return types[result];
        }
        result++;
        if (result >= weight_probs.size() - 1) {
            return types[weight_probs.size() - 1];
        }
    }
    return types[result];
}

enum State {NEUTRAL = 0, INFECTED = 1, VACCINATED = 2, CURED = 3};

struct AgentData {
    int unique_id;
    int n_inf_posts;
    double p_h;
    int timestep;
    AgentData(int id, int n_posts, double p, int t) : unique_id(id), n_inf_posts(n_posts), p_h(p), timestep(t) {}
};

struct MyAgent {

    int unique_id;
    int avg_node_degree;
    std::vector<double> weight_probs;

    std::vector<float> p_samples_reshare;
    std::vector<float> p_samples_influence;
    std::vector<float> p_samples_online;
    std::vector<float> p_samples_reject;
    std::vector<float> distribution_beliefs;
    std::vector<int> n_posts;

    std::vector<MyAgent> agent_list;
    std::vector<MyAgent *> neighbor_objects;
    int state;
    int time_step;
    bool influential;
    int belief_purity;
    int t_post;
    int pruning_var;
    int retweets;
    int pos;
    double mu_own;
    double sigma_own;
    double p_h;
    double p_h_initial;
    int n_neighbors;
    std::vector<int> neighbor_nodes;
    int n_inf_posts;
};

struct compare_agents {
        int node_degree;

        compare_agents(int degree) : node_degree(degree) {}

        bool operator()(MyAgent* a, MyAgent* b) {
            return (((a->n_neighbors / node_degree) + a->retweets) > ((b->n_neighbors / node_degree) + b->retweets));
        }
    };

struct compare_agents_beliefs {
    double p_h_val;
    compare_agents_beliefs(double val) : p_h_val(val) {}
    bool operator()(const MyAgent* a, const MyAgent* b) const {
        return std::abs(a->p_h - p_h_val) < std::abs(b->p_h - p_h_val);
    }
};

struct compare_agents_time {
    bool operator()(MyAgent* a, MyAgent* b) {
        return a->t_post > b->t_post;
    }
};

// Get curated neighbors
std::vector<MyAgent*> get_curated_neighbors(std::vector<int> n_posts, std::vector<MyAgent *> neighbor_objects, double p_h,
                                            int avg_node_degree, std::vector<double> weight_probs)
{
    int N = n_posts[rand() % n_posts.size()];

    std::vector<MyAgent*> neighbors = neighbor_objects;

    N = fmin(N, int(neighbors.size()));

    std::vector<MyAgent*> sorted_neighbors = neighbors;

    std::sort(sorted_neighbors.begin(), sorted_neighbors.end(),
              compare_agents_time());

    std::vector<MyAgent*> curated_neighbors_chrono;
    for (int i = 0; i < N; i++) {
        curated_neighbors_chrono.push_back(sorted_neighbors[i]);
    }

    sorted_neighbors = neighbors;
    double p_h_val = p_h;

    std::sort(sorted_neighbors.begin(), sorted_neighbors.end(), compare_agents_beliefs(p_h_val));


    std::vector<MyAgent*> curated_neighbors_belief;
    for (int i = 0; i < N; i++) {
        curated_neighbors_belief.push_back(sorted_neighbors[i]);
    }

    sorted_neighbors = neighbors;

    // Seed the random number generator
    std::srand(std::time(nullptr));
    std::random_shuffle(sorted_neighbors.begin(), sorted_neighbors.end());

    std::vector<MyAgent*> curated_neighbors_random;
    for (int i = 0; i < N; i++) {
        curated_neighbors_random.push_back(sorted_neighbors[i]);
    }

    sorted_neighbors = neighbors;
    double node_degree = avg_node_degree;

    std::sort(sorted_neighbors.begin(), sorted_neighbors.end(), compare_agents(node_degree));

    std::vector<MyAgent*> curated_neighbors_pop;
    for (int i = 0; i < N; i++) {
        curated_neighbors_pop.push_back(sorted_neighbors[i]);
    }

    std::vector<MyAgent*> curated_neighbors;
    std::vector<std::string> types;
    types.push_back("chrono");
    types.push_back("pop");
    types.push_back("belief");
    types.push_back("random");

    for (int n = 0; n < N; n++) {
        std::string curation_type =
                get_curation_type(weight_probs, types);
        if (curation_type == "chrono") {
            curated_neighbors.push_back(curated_neighbors_chrono[n]);
        } else if (curation_type == "pop") {
            curated_neighbors.push_back(curated_neighbors_pop[n]);
        } else if (curation_type == "belief") {
            curated_neighbors.push_back(curated_neighbors_belief[n]);
        } else if (curation_type == "random") {
            curated_neighbors.push_back(curated_neighbors_random[n]);
        }
    }
    return curated_neighbors;
}


// Agent Step
void step(MyAgent& a, int time_step) {

    // recalculates states each timestep
    double p_online = a.p_samples_online[rand() % a.p_samples_online.size()];
    float r = dis(gen);
    if (r < p_online) {
        a.t_post = time_step;
        if (a.state == INFECTED) {
            a.n_inf_posts = 1;
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start_reshare = std::chrono::high_resolution_clock::now();

    if (a.state == INFECTED) {
        p_online = a.p_samples_online[rand() % a.p_samples_online.size()];
        r = dis(gen);

        if (r < p_online) {
            // Cure Function (Removed from Agent Class)
            double p_reject = a.p_samples_reject[rand() % a.p_samples_reject.size()];
            r = dis(gen);
            if (r < p_reject) {
                a.state = CURED;
                a.n_inf_posts = 0;
            }

        } else {
            a.n_inf_posts = 0;
        }
    }

    if (a.state == NEUTRAL) {
        p_online = a.p_samples_online[rand() % a.p_samples_online.size()];
        r = dis(gen);
        if (r < p_online) {

            //  get curated neighbours
            std::vector<MyAgent*> curated_neighbors = get_curated_neighbors(a.n_posts, a.neighbor_objects, a.p_h,
                                                                            a.avg_node_degree, a.weight_probs);

            // check polarisation
            std::vector<double> belief_diff;
            for (size_t i = 0; i < curated_neighbors.size(); ++i) {
                MyAgent* &n = curated_neighbors[i];
                belief_diff.push_back(std::abs(n->p_h - a.p_h));
            }


            if (belief_diff.size() > 0 || std::accumulate(begin(belief_diff), end(belief_diff), 0) > 0) {
                a.belief_purity = std::accumulate(begin(belief_diff), end(belief_diff), 0) / belief_diff.size();
            } else {
                a.belief_purity = 1.0;
            }

            // Reshare Function (from Agent Class)

            // Sample p-reshare from distribution
            double p_reshare_orig = a.p_samples_reshare[rand() % a.p_samples_reshare.size()];
            double p_reshare;

            // Sample p-reject from distribution
            double p_reject = a.p_samples_reject[rand() % a.p_samples_reject.size()];


            for (size_t i = 0; i < curated_neighbors.size(); ++i) {
                MyAgent* &n = curated_neighbors[i];
                p_reshare = p_reshare_orig;
                if (n->influential == true){
                    double p_influence = a.p_samples_influence[rand() % a.p_samples_influence.size()];
                    p_reshare += p_influence;
                }
                if (n->state == VACCINATED) {
                    r = dis(gen);
                    if (r < p_reshare) {
                        double X = n->p_h;
                        n->p_h = bayesian_belief_updating(X, a.p_h, a.sigma_own, a.mu_own);
                        n->state = VACCINATED;
                        n->retweets++;
                        a.t_post = a.time_step;
                    }
                    if (r < p_reject) {
                        double X = n->p_h;
                        a.p_h = bayesian_belief_updating(X, a.p_h, a.sigma_own, a.mu_own);
                        a.state = CURED;
                        n->retweets++;
                        a.t_post = time_step;
                    }
                } else if (n->state == NEUTRAL) {
                    r = dis(gen);
                    if (r < p_reshare) {
                        double X = n->p_h;
                        a.p_h = bayesian_belief_updating(X, a.p_h, a.sigma_own, a.mu_own);
                        a.t_post = a.time_step;
                        n->retweets++;
                    }
                } else if (n->state == INFECTED) {
                    r = dis(gen);
                    if (r < p_reshare) {
                        double X = n->p_h;
                        a.p_h = bayesian_belief_updating(X, a.p_h, a.sigma_own, a.mu_own);
                        a.state = INFECTED;
                        n->retweets++;
                        a.t_post = a.time_step;
                        a.n_inf_posts = 1;
                    } else if (r < p_reject) {
                        double X = n->p_h;
                        a.p_h = bayesian_belief_updating(X, a.p_h, a.sigma_own, a.mu_own);
                        n->retweets++;
                        a.state = CURED;
                        a.t_post = a.time_step;
                    }
                }
            }

        }
    }

}



struct ThreadArgs {
    std::vector<MyAgent> agent_list;
    int thread_id;
    int start_idx;
    int end_idx;
    int t;

    std::vector<std::vector<AgentData> > *local_inf_dicts;

};

void *agent_step(void *thread_args) {

    ThreadArgs *args = (ThreadArgs *)thread_args;
    std::vector<AgentData> local_inf_dict;


    for (int j = args->start_idx; j < args->end_idx; j++) {

        MyAgent& a = args->agent_list[j];
        if (args->t > 0){
            step(a, args->t);
        }

        a.time_step = args->t;

        local_inf_dict.push_back(AgentData(a.unique_id, a.n_inf_posts, a.p_h, a.time_step));


    }

    args->local_inf_dicts->push_back(local_inf_dict);

    pthread_exit(NULL);
}

double average(std::vector<double> v) {
    double sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum / v.size();
}

double run_abm(
        int n_threads,
        int pop,
        int neighbors,
        int timesteps,
        float initially_influential_prop,
        float initially_infected_prop,
        float initially_vaccinated_prop,
        std::vector<float> p_samples_reshare,
        std::vector<float> p_samples_reject,
        std::vector<float> p_samples_online,
        float weight_chrono,
        float weight_belief,
        float weight_pop,
        float weight_random,
        std::map<std::string, std::vector<float> > df_hours) {

    // Parameters
    int num_nodes = pop;
    int avg_node_degree = neighbors;
    std::vector<float> distribution_beliefs;
    distribution_beliefs.push_back(0.01);
    std::vector<std::vector<AgentData> > datacollector;

    std::vector<double> weight_probs;
    weight_probs.push_back(weight_chrono);
    weight_probs.push_back(weight_pop);
    weight_probs.push_back(weight_belief);
    weight_probs.push_back(weight_random);

    // Initializing graph using Barabasi-Albert model

    std::vector<std::vector<int> > BA;
    for (int i = 0; i < num_nodes; i++)
    {
        BA.push_back(std::vector<int>());
    }

    // Adding edges
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = i+1; j < num_nodes; j++)
        {
            // Generating a random number
            int random_number = rand() % num_nodes;
            // Checking if it's less than m
            if (random_number < avg_node_degree)
            {
                BA[i].push_back(j);
            }
        }
    }

    int n_inf = 0;
    int steps = 0;
    int time_step = 0;

    std::vector<MyAgent> agent_list;

    for (int i = 0; i < num_nodes; i++) {

        MyAgent a;
        a.unique_id = i;
        a.avg_node_degree = neighbors;
        a.weight_probs = weight_probs;
        a.distribution_beliefs = distribution_beliefs;

        a.p_samples_reshare = p_samples_reshare;
        a.p_samples_influence.push_back(0.1);
        a.p_samples_online = p_samples_online;
        a.p_samples_reject = p_samples_reject;
        a.n_posts.push_back(40);
        a.state = NEUTRAL;
        a.time_step = 0;
        a.influential = false;
        a.belief_purity = 0;
        a.t_post = 0;
        a.pruning_var = 0;
        a.retweets = 0;
        a.mu_own = 0;
        a.sigma_own = 0;
        a.p_h = 0;
        a.p_h_initial = 0;
        a.n_inf_posts = 0;

        a.pos = i;
        a.neighbor_nodes = BA[a.pos];
        a.n_neighbors = a.neighbor_nodes.size();

        // Set the random engine
        std::default_random_engine random_engine;

        // Set the vaccinated
        double r = dis(gen);
        if (r < initially_vaccinated_prop){
            a.state = VACCINATED;
        }

        // Set the influential
        r = dis(gen);
        if (r < initially_influential_prop){
            a.influential = true;
        }

        // Set the infected
        r = dis(gen);
        if (r < initially_infected_prop){
            a.state = INFECTED;
            n_inf = n_inf + 1;
            a.n_inf_posts = 1;
        }

        // Generate three pieces of evidence
        std::vector<int> evidence(3);

        std::vector<int> weights;
        weights.push_back(1);
        weights.push_back(1);

        std::discrete_distribution<int> evidence_distribution(weights.begin(), weights.end());

        for (int i = 0; i < evidence.size(); i++) {
            evidence[i] = evidence_distribution(random_engine);
        }

        // Calculate the mean of the vector
        a.mu_own = 0;
        for (int i = 0; i < evidence.size(); i++) {
            a.mu_own += evidence[i];
        }
        a.mu_own = a.mu_own / evidence.size();

        // Calculate the standard deviation of the vector
        a.sigma_own = 0;
        for (int i = 0; i < evidence.size(); i++) {
            a.sigma_own += (evidence[i] - a.mu_own) * (evidence[i] - a.mu_own);
        }
        a.sigma_own = a.sigma_own / evidence.size();
        a.sigma_own = sqrt(a.sigma_own);

        std::srand(std::time(0)); // use current time as seed for random generator
        int random_pos = std::rand() %
                         distribution_beliefs.size();  // Modulo to restrict the number of random values to be at most A.size()-1
        a.p_h_initial = distribution_beliefs[random_pos];
        a.p_h = a.p_h_initial;

        agent_list.push_back(a);

    }


    //    Connect agents to neighbours
    for (int j = 0; j < agent_list.size(); j++) {
        MyAgent& a = agent_list[j];

        std::vector<MyAgent*> neighbor_objects; // vector of pointers to MyAgent objects

        for (std::vector<int>::const_iterator it = a.neighbor_nodes.begin(); it != a.neighbor_nodes.end(); ++it) {
            const int &i = *it;
            neighbor_objects.push_back(&agent_list[i]); // add pointer to MyAgent object in agent_list
        }

        a.neighbor_objects = neighbor_objects;
    }

    // Run simulation and multi-threading
    std::vector<double> thread_creation_time;
    std::vector<double> simulation_time;

    std::vector<pthread_t> threads(n_threads);
    std::vector<ThreadArgs> args(n_threads);

    for (int i = 0; i < n_threads; i++) {
        args[i].agent_list = agent_list;
        args[i].thread_id = i;
    }

    for (int t = 0; t < timesteps; t++) {
        std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> start_t = std::chrono::high_resolution_clock::now();

        std::vector<AgentData> inf_dict;
        std::vector<std::vector<AgentData> >& local_inf_dicts = *(args[0].local_inf_dicts);

        int half_size = agent_list.size() / n_threads;

        for (int i = 0; i < n_threads; i++) {
            args[i].t = t;
            args[i].start_idx = i * half_size;
            args[i].end_idx = (i + 1) * half_size;
            args[i].local_inf_dicts = new std::vector<std::vector<AgentData> >(timesteps);
            if (i == n_threads - 1) {
                args[i].end_idx = agent_list.size();
            }
        }


        for (int i = 0; i < n_threads; i++) {
            pthread_create(&threads[i], NULL, agent_step, (void*)&args[i]);
        }

        for (int i = 0; i < n_threads; i++) {
            pthread_join(threads[i], NULL);
        }


        // merge local inf dicts
        for (size_t i = 0; i < n_threads; ++i) {
            std::vector<std::vector<AgentData> >& local_inf_dicts = *args[i].local_inf_dicts;
            for (size_t j = 0; j < local_inf_dicts.size(); ++j) {
                const std::vector<AgentData>& local_inf_dict = local_inf_dicts[j];
                inf_dict.insert(inf_dict.end(), local_inf_dict.begin(), local_inf_dict.end());
            }
        }


        datacollector.push_back(inf_dict);

        steps = steps + 1;
        time_step = time_step + 1;

    }

    // Clean up thread-related resources
    for (int i = 0; i < n_threads; i++) {
        delete args[i].local_inf_dicts;
    }

    // Post-processing
    std::vector<int> agent_state(datacollector.size(), 0);

    for (int i = 0; i < datacollector.size(); i++) {
        int no_inf_agents = 0;
        for (int j = 0; j < datacollector[i].size(); j++) {
            no_inf_agents += datacollector[i][j].n_inf_posts;
        }
        agent_state[i] = no_inf_agents;
    }

    std::vector<std::map<std::string, float> > total_data;

    for (int i = 0; i < agent_state.size(); i++) {
        float prop;
        if (agent_state[i] > 0) {
            float as = agent_state[i];
            prop = as / num_nodes;
        }
        else {
            prop = 0;
        }

        std::map<std::string, float> data;
        data["steps"] = static_cast<float>(i);
        data["prop_tweets_inf_abm"] = prop;

        total_data.push_back(data);
    }


    std::vector<std::map<std::string, float> > abm_output = total_data;
    std::vector<std::map<std::string, float> > output;


    for (int i = 0; i < abm_output.size(); i++) {
        for (int j = 0; j < df_hours["steps"].size(); j++) {
            if (abm_output[i]["steps"] == df_hours["steps"][j]) {
                std::map<std::string, float> data;
                data["steps"] = abm_output[i]["steps"];
                data["prop_tweets_inf_rw"] = df_hours["prop_tweets_inf_rw"][j];
                data["prop_tweets_inf_abm"] = abm_output[i]["prop_tweets_inf_abm"];
                output.push_back(data);
            }
        }
    }


    double distance_updated = 0;

    for(int i = 0; i < output.size(); i++)
    {
        distance_updated += pow(output[i]["prop_tweets_inf_rw"] - output[i]["prop_tweets_inf_abm"], 2);
    }

    distance_updated = sqrt(distance_updated);

    return distance_updated;
}

int main() {
    std::vector<double> time_list;

    // random seed
    std::mt19937 random(20);
    std::mt19937 np_random(21);

    // parameters
    int n_threads;
    int pop;
    int neighbors;
    int timesteps;
    float initially_influential_prop;
    float initially_infected_prop;
    float initially_vaccinated_prop;
    float preshare_mean;
    float preject_mean;
    float ponline_mean;
    std::string filename;

    // define weights
    float weight_chrono;
    float weight_belief;
    float weight_pop;
    float weight_random;

    std::cin >> n_threads;
    std::cin >> pop;
    std::cin >> neighbors;
    std::cin >> timesteps;
    std::cin >> initially_influential_prop;
    std::cin >> initially_infected_prop;
    std::cin >> initially_vaccinated_prop;
    std::cin >> preshare_mean;
    std::cin >> preject_mean;
    std::cin >> ponline_mean;
    std::cin >> weight_chrono;
    std::cin >> weight_belief;
    std::cin >> weight_pop;
    std::cin >> weight_random;
    std::cin >> filename;
    std::map<std::string, std::vector<float> > df_hours = readCSV(filename);

    int n = 100;
    std::vector<float> p_samples_reshare;
    std::normal_distribution<double> dist_reshare(preshare_mean, preshare_mean/2);
    for (int i = 0; i < n; i++) {
        double num = std::max(0.0, dist_reshare(gen));
        p_samples_reshare.push_back(num);
    }

    std::vector<float> p_samples_reject;
    std::normal_distribution<double> dist_reject(preject_mean, preject_mean/2);
    for (int i = 0; i < n; i++) {
        double num = std::max(0.0, dist_reject(gen));
        p_samples_reject.push_back(num);
    }

    std::vector<float> p_samples_online;
    std::normal_distribution<double> dist_online(ponline_mean, ponline_mean/2);
    for (int i = 0; i < n; i++) {
        double num = std::max(0.0, dist_online(gen));
        p_samples_online.push_back(num);
    }

    double distance = run_abm(
            n_threads,
            pop,
            neighbors,
            timesteps,
            initially_influential_prop,
            initially_infected_prop,
            initially_vaccinated_prop,
            p_samples_reshare,
            p_samples_reject,
            p_samples_online,
            weight_chrono,
            weight_belief,
            weight_pop,
            weight_random,
            df_hours);

    std::cout << distance << std::endl;

    return 0;
}
