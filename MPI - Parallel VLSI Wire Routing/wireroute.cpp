#include "wireroute.h"
// #include "mpi.h"
#include "helpers.h"
#include "mpi/mpi.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <libgen.h>

// Perform computation, including reading/writing output files
void compute(int procID, int nproc, char *inputFilename, double SA_prob, int SA_iters) {
    // TODO Implement code here
    // TODO Decide which processors should be reading/writing files

    srand(time(nullptr));

    const int root = 0; // Set the rank 0 process as the root process

    int dim_x, dim_y, num_of_wires;

    // Initialize inputs
    std::ifstream input_file;
    if (procID == root) {
        input_file.open(inputFilename);
        if (!input_file.is_open()) {
            printf("Unable to open file: %s.\n", inputFilename);
            return;
        }

        input_file >> dim_x >> dim_y >> num_of_wires;
    }
    MPI_Bcast(&dim_x, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&dim_y, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&num_of_wires, 1, MPI_INT, root, MPI_COMM_WORLD);

    Wire *wires = (Wire *)calloc(num_of_wires, sizeof(Wire));
    Route *routes = (Route *)calloc(num_of_wires, sizeof(Route));
    cost_t *costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    const size_t costs_size = dim_x * dim_y * sizeof(cost_t);

    Data data = {dim_x, num_of_wires, wires};
    // Result result = {costs, prev_routes};

    int work_per_proc[128];
    uint64_t workload_per_proc[128];

    if (procID == root) {
        uint64_t total_computation_cost = 0;
        for (int i = 0; i < num_of_wires; ++i) {
            int start_x, start_y, end_x, end_y;
            input_file >> start_x >> start_y >> end_x >> end_y;
            wires[i] = {{start_x, start_y}, {end_x, end_y}, i};
            total_computation_cost += wires[i].computation_cost;
        }
        input_file.close();

        routes = (Route *)calloc(num_of_wires, sizeof(Route));
        for (int wire_id = 0; wire_id < num_of_wires; wire_id++) {
            routes[wire_id] = generate_random_route(wires[wire_id]);
        }

        costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
        data = {dim_x, num_of_wires, wires};
        // result = {costs, routes};
        auto total_metrics = walk_all_routes(data, costs, routes, 1);
        // std::cout << total_metrics << std::endl;
        // std::cout << total_computation_cost << std::endl;

        // load-balancing
        uint64_t computation_per_proc = total_computation_cost / nproc;
        // std::cout << computation_per_proc << std::endl;
        uint64_t accumulated_computation = 0;
        int proc_id = 0;
        // std::cout << nproc << std::endl;
        for (int wire_id = 0; wire_id < num_of_wires; ++wire_id) {
            accumulated_computation += wires[wire_id].computation_cost;

            // last thread
            if (wire_id == num_of_wires - 1) {
                work_per_proc[proc_id] = wire_id + 1;
                workload_per_proc[proc_id] = accumulated_computation;
            } else if (accumulated_computation >= computation_per_proc * 0.999 && proc_id != nproc - 1) {
                assert(proc_id < nproc);
                work_per_proc[proc_id] = wire_id + 1;
                workload_per_proc[proc_id] = accumulated_computation;
                proc_id += 1;
                accumulated_computation = 0;
            }
        }

        // for (int i = 0; i != nproc; ++i) {
        //     std::cout << i << ": " << work_per_proc[i] << "(" << (double)workload_per_proc[i] / (double)computation_per_proc << ")\n";
        // }
    }
    MPI_Bcast(wires, num_of_wires * sizeof(Wire), MPI_BYTE, root, MPI_COMM_WORLD);
    MPI_Bcast(routes, num_of_wires * sizeof(Route), MPI_BYTE, root, MPI_COMM_WORLD);
    // std::cout << proc_info() << wires[1111].computation_cost << " " << wires[100].computation_cost << " " << num_of_wires << std::endl;
    MPI_Bcast(costs, costs_size, MPI_BYTE, root, MPI_COMM_WORLD);
    MPI_Bcast(work_per_proc, nproc, MPI_INT, root, MPI_COMM_WORLD);

    cost_t *new_costs = (cost_t *)calloc(dim_x * dim_y, sizeof(cost_t));
    memcpy(new_costs, costs, costs_size);

    int wire_id_begin = (procID == 0 ? 0 : work_per_proc[procID - 1]);
    int wire_id_end = work_per_proc[procID];

    // std::deque<Wire> work_queue;
    // for (int wire_id = wire_id_begin; wire_id < wire_id_end; ++wire_id) {
    //     work_queue.push_back(wires[wire_id]);
    // }
    // std::cout << proc_info() << wire_id_begin << " " << wire_id_end << std::endl;

    for (int iter_id = 0; iter_id != SA_iters; ++iter_id) {
        for (int wire_id = wire_id_begin; wire_id < wire_id_end; ++wire_id) {
            Wire wire = wires[wire_id];
            Route prev_route = routes[wire_id];
            prev_route.metrics = walk_a_route(data, new_costs, prev_route, -1);

            if (is_random_route(SA_prob)) {
                Route new_route = generate_random_route(wire);
                walk_a_route(data, new_costs, new_route, 1);
                routes[wire_id] = new_route;
                continue;
            }

            std::vector<Route> all_routes_for_one_wire = generate_routes(wire);
            Route best_route = prev_route;
            for (int route_id = 0; route_id != all_routes_for_one_wire.size(); ++route_id) {
                Route new_route = all_routes_for_one_wire[route_id];
                new_route.metrics = walk_a_route(data, new_costs, new_route, 0);
                if (new_route.metrics < best_route.metrics) {
                    best_route = new_route;
                }
            }
            best_route.metrics = walk_a_route(data, new_costs, best_route, 1);
            routes[wire_id] = best_route;
        }
    }

    Metrics metrics_all_routes = walk_all_routes(data, new_costs, routes, 0);
    std::cout << metrics_all_routes << std::endl;

    // write to file
    if (procID == root) {
        std::string filename = std::string(basename(inputFilename));
        std::string name = filename.substr(0, filename.size() - 4);

        // Write costs
        std::ofstream costs_file("cost_" + name + "_" + std::to_string(nproc) + ".txt");
        costs_file << dim_x << " " << dim_y << "\n";
        for (int i = 0; i != dim_x * dim_y; ++i) {
            costs_file << (uint32_t)new_costs[i] << ((i % dim_y == dim_y - 1) ? "\n" : " ");
        }
        costs_file.close();

        // Write wires
        std::ofstream wires_file("output_" + name + "_" + std::to_string(nproc) + ".txt");
        wires_file << dim_x << " " << dim_y << "\n"
                   << num_of_wires << "\n";
        for (int i = 0; i != num_of_wires; ++i)
            wires_file << routes[i] << "\n";
        wires_file.close();
    }
}
