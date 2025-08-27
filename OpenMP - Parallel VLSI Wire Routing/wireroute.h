/**
 * Parallel VLSI Wire Routing via OpenMP
 */

#ifndef __WIREOPT_H__
#define __WIREOPT_H__

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

/* Define the data structure for wire here */

struct Point {
    int x, y;
};
inline bool operator==(const Point &lhs, const Point &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
inline bool operator!=(const Point &lhs, const Point &rhs) { return !(lhs == rhs); }

inline int signX(const Point &p1, const Point &p2) { return p2.x >= p1.x ? +1 : -1; }
inline int signY(const Point &p1, const Point &p2) { return p2.y >= p1.y ? +1 : -1; }

struct Wire {
    Point start, end;
    inline int signX() { return ::signX(start, end); }
    inline int signY() { return ::signY(start, end); }
};

typedef int cost_t;
#define MAX_COST INT32_MAX

struct Data {
    int dim_x, dim_y;
    int num_of_wires;
    Wire *wires;
    int num_of_threads;
    double SA_prob;
    int SA_iters;
};

struct Metrics {
    cost_t max_cost_value;
    cost_t sum_cost_values;

    Metrics(cost_t max_cost_value = 0, cost_t sum_cost_values = 0) : max_cost_value(max_cost_value), sum_cost_values(sum_cost_values) {}

    void update(cost_t new_cost) {
        this->max_cost_value = std::max(this->max_cost_value, new_cost);
        this->sum_cost_values += new_cost;
    }

    void update(Metrics new_metrics) {
        this->max_cost_value = std::max(this->max_cost_value, new_metrics.max_cost_value);
        this->sum_cost_values += new_metrics.sum_cost_values;
    }
};

struct Route {
    Wire wire;
    Point p1, p2;
    Metrics metrics = Metrics{MAX_COST, MAX_COST};
    Route() {}
    explicit Route(Wire wire) : wire(wire) {}
};

struct Result {
    cost_t *costs;
    Route *routes;
};

bool operator<(const Metrics &lhs, const Metrics &rhs) {
    if (lhs.max_cost_value != rhs.max_cost_value)
        return lhs.max_cost_value < rhs.max_cost_value;
    return lhs.sum_cost_values < rhs.sum_cost_values;
}

std::ostream &operator<<(std::ostream &os, const Metrics &metrics) {
    os << "Max cost: " << metrics.max_cost_value << ", Sum cost: " << metrics.sum_cost_values << "";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << p.x << " " << p.y;
    return os;
}

std::ostream &operator<<(std::ostream &os, const Wire &wire) {
    os << wire.start << " " << wire.end;
    return os;
}

std::ostream &operator<<(std::ostream &os, const Route &route) {
    os << route.wire.start << " ";
    if (route.p1 != route.wire.start)
        os << route.p1 << " ";
    if (route.p2 != route.p1)
        os << route.p2 << " ";
    os << route.wire.end;
    return os;
}

const char *get_option_string(const char *option_name,
                              const char *default_value);
int get_option_int(const char *option_name, int default_value);
float get_option_float(const char *option_name, float default_value);

Metrics walk_a_line(Data data, Result result, Point p1, Point p2, int cost_change);
Metrics walk_a_route(Data data, Result result, Route route, int cost_change);
Metrics walk_all_routes(Data data, Result result, int cost_change);

void wire_routing(Data data, Result result, const std::vector<std::vector<Route>> &possible_routes);
void wire_routing_sequential(Data data, Result result, const std::vector<std::vector<Route>> &possible_routes);
void solve_all_metrics(Data data, Result result, const std::vector<Route> &routes);

std::vector<std::vector<Route>> prepare_all_routes(Data data, Result result);
std::vector<Route> prepare_all_routes_flatten(Data data, Result result);

Route generate_random_route(Data input_data, Wire wire);

#endif
