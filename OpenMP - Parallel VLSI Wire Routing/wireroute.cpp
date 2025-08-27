/**
 * Parallel VLSI Wire Routing via OpenMP
 */

#include "wireroute.h"

#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name,
                              const char *default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return _argv[i + 1];
    return default_value;
}

int get_option_int(const char *option_name, int default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return atoi(_argv[i + 1]);
    return default_value;
}

float get_option_float(const char *option_name, float default_value) {
    for (int i = _argc - 2; i >= 0; i -= 2)
        if (strcmp(_argv[i], option_name) == 0)
            return (float)atof(_argv[i + 1]);
    return default_value;
}

static void show_help(const char *program_path) {
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}

int main(int argc, const char *argv[]) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto init_start = Clock::now();
    double init_time = 0;

    _argc = argc - 1;
    _argv = argv + 1;

    const char *input_filename = get_option_string("-f", NULL);
    int num_of_threads = get_option_int("-n", 1);
    double SA_prob = get_option_float("-p", 0.1f);
    int SA_iters = get_option_int("-i", 5);

    int error = 0;

    if (input_filename == NULL) {
        printf("Error: You need to specify -f.\n");
        error = 1;
    }

    if (error) {
        show_help(argv[0]);
        return 1;
    }

    printf("Number of threads: \t\t\t[%d]\n", num_of_threads);
    // printf("Probability parameter for simulated annealing: %lf.\n", SA_prob);
    // printf("Number of simulated annealing iterations: %d\n", SA_iters);
    // printf("Input file: %s\n", input_filename);

    FILE *input = fopen(input_filename, "r");

    if (!input) {
        printf("Unable to open file: %s.\n", input_filename);
        return 1;
    }

    int dim_x, dim_y;
    int num_of_wires;

    fscanf(input, "%d %d\n", &dim_x, &dim_y);
    fscanf(input, "%d\n", &num_of_wires);

    srand(time(nullptr));
    omp_set_num_threads(num_of_threads);
    omp_set_nested(1);

    Wire *wires = (Wire *)calloc(num_of_wires, sizeof(Wire));
    /* Read the grid dimenseon and wire information from file */
    for (int i = 0; i < num_of_wires; i++) {
        fscanf(input, "%d %d %d %d\n", &wires[i].start.x, &wires[i].start.y, &wires[i].end.x, &wires[i].end.y);
    }

    Data data = {
        dim_x, dim_y,
        num_of_wires,
        wires,
        num_of_threads,
        SA_prob,
        SA_iters};

    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    /* Initialize cost matrix */
    /* Initailize additional data structures needed in the algorithm */
    Route *routes = (Route *)calloc(num_of_wires, sizeof(Route));

    for (int i = 0; i < num_of_wires; i++) {
        routes[i] = generate_random_route(data, wires[i]);
    }
    Result result = {costs, routes};
    walk_all_routes(data, result, 1);

    std::vector<std::vector<Route>> possible_routes = prepare_all_routes(data, result);
    std::vector<Route> possible_routes_flatten = prepare_all_routes_flatten(data, result);

    // std::cout << "Total routes: " << possible_routes.size() << std::endl;
    // std::cout << "Sizes: ";
    // for (auto routes : possible_routes) {
    //     std::cout << routes.size() << " ";
    // }
    // std::cout << std::endl;

    init_time += duration_cast<dsec>(Clock::now() - init_start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto compute_start = Clock::now();
    double compute_time = 0;

    /**
   * Implement the wire routing algorithm here
   * Feel free to structure the algorithm into different functions
   * Don't use global variables.
   * Use OpenMP to parallelize the algorithm.
   */
    for (int i = 0; i != SA_iters; ++i) {
        wire_routing(data, result, possible_routes);
        // wire_routing_sequential(data, result, possible_routes);
        // solve_all_metrics(data, result, possible_routes_flatten);
    }

    compute_time += duration_cast<dsec>(Clock::now() - compute_start).count();
    printf("Computation Time: \t\t\t[%lf].\n", compute_time);

    /* Write wires and costs to files */
    // Print metrics to screen
    Metrics metrics_all_routes = walk_all_routes(data, result, 0);
    std::cout << metrics_all_routes << std::endl;

    // Write costs
    std::ofstream costs_file("output_" + std::to_string(num_of_threads) + ".txt");
    costs_file << dim_x << " " << dim_y << "\n";
    for (int i = 0; i != dim_x * dim_y; ++i) {
        costs_file << costs[i] << ((i % dim_y == dim_y - 1) ? "\n" : " ");
    }
    costs_file.close();

    // Write wires
    std::ofstream wires_file("wires.txt");
    wires_file << dim_x << " " << dim_y << "\n"
               << num_of_wires << "\n";
    for (int i = 0; i != num_of_wires; ++i)
        wires_file << routes[i] << "\n";
    wires_file.close();

    return 0;
}

Metrics walk_a_line(Data data, Result result, Point p1, Point p2, int cost_change) {
    assert(p1.x == p2.x || p1.y == p2.y);

    Metrics metrics_of_line;

    if (p1 == p2)
        return metrics_of_line;

    if (p1.x == p2.x) {
        for (int y = p1.y; y != p2.y; y += signY(p1, p2)) {
            int costsIdx = y * data.dim_x + p1.x;
            result.costs[costsIdx] += cost_change;
            metrics_of_line.update(result.costs[costsIdx]);
        }
    } else {
        for (int x = p1.x; x != p2.x; x += signX(p1, p2)) {
            int costsIdx = p1.y * data.dim_x + x;
            result.costs[costsIdx] += cost_change;
            metrics_of_line.update(result.costs[costsIdx]);
        }
    }
    return metrics_of_line;
}

Metrics walk_a_route(Data data, Result result, Route route, int cost_change) {
    Metrics metrics_of_route;

    metrics_of_route.update(walk_a_line(data, result, route.wire.start, route.p1, cost_change));
    metrics_of_route.update(walk_a_line(data, result, route.p1, route.p2, cost_change));
    metrics_of_route.update(walk_a_line(data, result, route.p2, route.wire.end, cost_change));
    // arrive at terminal point
    int costsIdx = route.wire.end.y * data.dim_x + route.wire.end.x;
    result.costs[costsIdx] += cost_change;
    metrics_of_route.update(result.costs[costsIdx]);

    return metrics_of_route;
}

Metrics walk_all_routes(Data data, Result result, int cost_change) {
    Metrics metrics_of_all_routes;
    for (int i = 0; i != data.num_of_wires; i++) {
        Route route = result.routes[i];
        metrics_of_all_routes.update(walk_a_route(data, result, route, cost_change));
    }
    return metrics_of_all_routes;
}

inline bool is_random_route(Data data) { return (rand() % 100) <= (data.SA_prob * 100); }

void solve_all_metrics(Data data, Result result, const std::vector<Route> &routes) {
    // std::vector<Route> routes;// = aroutes;
    size_t route_len = routes.size();

#pragma omp parallel for schedule(guided)
    for (size_t route_id = 0; route_id < route_len; ++route_id) {
        Route new_route = routes[route_id];
        // Metrics metrics_of_new_route =
        walk_a_route(data, result, new_route, 0);
    }
}

void wire_routing(Data data, Result result, const std::vector<std::vector<Route>> &possible_routes) {
    for (int wire_id = 0; wire_id < data.num_of_wires; wire_id++) {
        Wire wire = data.wires[wire_id];
        Route prev_route = result.routes[wire_id];
        prev_route.metrics = walk_a_route(data, result, prev_route, -1);

        // choose a random path
        if (is_random_route(data)) {
            Route route(wire);
            route = generate_random_route(data, wire);
            result.routes[wire_id] = route;
            walk_a_route(data, result, route, 1);
            continue;
        }

        const std::vector<Route> &routes = possible_routes[wire_id];

#pragma omp declare reduction(min_route:Route \
                              : omp_out = omp_in.metrics < omp_out.metrics ? omp_in : omp_out)

        size_t routes_len = routes.size();
        Route best_route; // Route() has max cost
#pragma omp parallel for schedule(guided) reduction(min_route \
                                                    : best_route)
        for (size_t route_id = 0; route_id < routes_len; ++route_id) {
            Route new_route = routes[route_id];
            new_route.metrics = walk_a_route(data, result, new_route, 0);
            if (new_route.metrics < best_route.metrics)
                best_route = new_route;
        }

        walk_a_route(data, result, best_route, 1);
        result.routes[wire_id] = best_route;
    }
}

std::vector<std::vector<Route>> prepare_all_routes(Data data, Result result) {
    std::vector<std::vector<Route>> possible_routes(data.num_of_wires);

    for (int wire_id = 0; wire_id != data.num_of_wires; wire_id++) {
        Wire wire = data.wires[wire_id];

        std::vector<Route> routes;
        routes.reserve(abs(wire.end.y - wire.start.y) + abs(wire.end.x - wire.start.x));

        for (int y = wire.start.y; y != wire.end.y; y += wire.signY()) {
            Route route(wire);
            route.p1 = {wire.start.x, y};
            route.p2 = {wire.end.x, y};
            routes.push_back(route);
        }

        for (int x = wire.start.x; x != wire.end.x; x += wire.signX()) {
            Route route(wire);
            route.p1 = {x, wire.start.y};
            route.p2 = {x, wire.end.y};
            routes.push_back(route);
        }
        possible_routes[wire_id] = routes;
    }
    return possible_routes;
}

std::vector<Route> prepare_all_routes_flatten(Data data, Result result) {
    std::vector<Route> routes;

    for (int wire_id = 0; wire_id != data.num_of_wires; wire_id++) {
        Wire wire = data.wires[wire_id];

        routes.reserve(routes.size() + abs(wire.end.y - wire.start.y) + abs(wire.end.x - wire.start.x));

        for (int y = wire.start.y; y != wire.end.y; y += wire.signY()) {
            Route route(wire);
            route.p1 = {wire.start.x, y};
            route.p2 = {wire.end.x, y};
            routes.push_back(route);
        }

        for (int x = wire.start.x; x != wire.end.x; x += wire.signX()) {
            Route route(wire);
            route.p1 = {x, wire.start.y};
            route.p2 = {x, wire.end.y};
            routes.push_back(route);
        }
    }
    return routes;
}

Route generate_random_route(Data data, Wire wire) {
    Route route(wire);

    int dx = wire.end.x - wire.start.x;
    int dy = wire.end.y - wire.start.y;

    bool vertical_first = rand() % 2;
    float p = (rand() % 100) / 100.f;

    if (vertical_first) {
        route.p1 = {wire.start.x, wire.start.y + int(p * dy)};
        route.p2 = {wire.end.x, route.p1.y};
    } else {
        route.p1 = {wire.start.x + int(p * dx), wire.start.y};
        route.p2 = {route.p1.x, wire.end.y};
    }
    return route;
}

void wire_routing_sequential(Data data, Result result, const std::vector<std::vector<Route>> &possible_routes) {
    for (int wire_id = 0; wire_id < data.num_of_wires; wire_id++) {
        Wire wire = data.wires[wire_id];
        Route prev_route = result.routes[wire_id];
        prev_route.metrics = walk_a_route(data, result, prev_route, -1);

        // choose a random path
        if (is_random_route(data)) {
            Route route(wire);
            route = generate_random_route(data, wire);
            result.routes[wire_id] = route;
            walk_a_route(data, result, route, 1);
            continue;
        }

        Route best_route; // Route() has max cost

        const std::vector<Route> &routes = possible_routes[wire_id];
        size_t routes_len = routes.size();

        for (size_t route_id = 0; route_id < routes_len; ++route_id) {
            Route new_route = routes[route_id];
            new_route.metrics = walk_a_route(data, result, new_route, 0);
            if (new_route.metrics < best_route.metrics) {
                best_route = new_route;
            }
        }

        walk_a_route(data, result, best_route, 1);
        result.routes[wire_id] = best_route;
    }
}