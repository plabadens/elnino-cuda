#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

using real_t = float;
constexpr size_t NX = 300, NY = 400;         // number of grid points
constexpr size_t XX = 3000000, YY = 4000000; // size of the grid 3000km x 4000km

real_t dx = XX / NX; // Integration step size in the horizontal direction
real_t dy = YY / NY;
using grid_t = std::array<std::array<real_t, NY>, NX>;
double dt = 100; // Size of the integration time step

class Sim_Configuration {
public:
  int iter = 100000;   // Number of iterations
  double dt = 1000;    // Size of the integration time step
  real_t g = 0.01;     // Gravitational acceleration
  real_t dx = XX / NX; // Integration step size in the horizontal direction
  real_t dy = YY / NY;
  int data_period = 1000; // how often to save coordinate to file
  std::string filename =
      "sw_output.data"; // name of the output file with history
  real_t tau_over_D = 0.003;

  Sim_Configuration(std::vector<std::string> argument) {
    for (long unsigned int i = 1; i < argument.size(); i += 2) {
      std::string arg = argument[i];
      if (arg == "-h") { // Write help
        std::cout << "./par --iter <number of iterations> --dt <time step>"
                  << " --g <gravitational const> --dx <x grid size> --dy <y "
                     "grid size>"
                  << "--fperiod <iterations between each save> --out <name of "
                     "output file>\n";
        exit(0);
      } else if (i == argument.size() - 1)
        throw std::invalid_argument("The last argument (" + arg +
                                    ") must have a value");
      else if (arg == "--iter") {
        if ((iter = std::stoi(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "iter most be a positive integer (e.g. -iter 1000)");
      } else if (arg == "--dt") {
        if ((dt = std::stod(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "dt most be a positive real number (e.g. -dt 0.05)");
      } else if (arg == "--g") {
        g = std::stod(argument[i + 1]);
      } else if (arg == "--dx") {
        if ((dx = std::stod(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "dx most be a positive real number (e.g. -dx 1)");
      } else if (arg == "--dy") {
        if ((dy = std::stod(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "dy most be a positive real number (e.g. -dy 1)");
      } else if (arg == "--fperiod") {
        if ((data_period = std::stoi(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "fperiod most be a positive integer (e.g. -fperiod 100)");
      } /*else if(arg=="--n_gangs"){
          if ((n_gangs = std::stoi(argument[i+1])) < 0)
              throw std::invalid_argument("n_gangs most be a positive integer
          (e.g. -n_gangs 14)");} */
      else if (arg == "--out") {
        filename = argument[i + 1];
      } else {
        std::cout << "---> error: the argument type is not recognized \n";
      }
    }
  }
};

class Water {
public:
  grid_t u{};                   // The speed in the horizontal direction.
  grid_t v{};                   // The speed in the vertical direction.
  grid_t e{};                   // The water elevation.
  std::array<double, NY + 2> f; // +2 for ghost lines
  double beta = 2e-11;
  Water() {
    for (size_t i = 1; i < NX - 1; ++i) {
      for (size_t j = 1; j < NY - 1; ++j) {
        real_t ii = 100.0 * (i - (NX - 100.0) / 2.0) / NX;
        real_t jj = 100.0 * (j - (NY - 2.0) / 2.0) / NY;
        e[i][j] = std::exp(
            -0.02 *
            (ii * ii + jj * jj)); // centered gaussian doesn't move w/ corriolis
                                  // e[i][j] = 0.1;
      }
    }

    // calculates f, scaling with grid size. Equator is off by one
    for (uint64_t i = 0; i < f.size(); i++) {
      f[i] = beta * (double((NY) / 2) - double(i)) * dy *dt *1000; // grid_size=10000. Additional multiplicative factor to enhance the effect
    }
  }
};

void integrate(Water &w, const real_t dt,
               const real_t dx, const real_t dy, const real_t g) {

  for (uint64_t i = 0; i < NX; i++) {
    for (uint64_t j = 0; j < NY; j++) {
      real_t wu_copy = w.u[i][j];
      w.u[i][j] -=
          dt / dx * g *
          (w.e[i + 1][j] - w.e[i][j] + w.f[j] * w.v[i][j]); //+ tau_over_D);
      w.v[i][j] -= dt / dy * g * (w.e[i][j + 1] - w.e[i][j] - w.f[j] * wu_copy);
    }
  }

  // fix boundaries for u
  for (uint64_t j = 0; j < NY; j++) {
    w.u[0][j] = 0;
    w.u[NX - 1][j] = 0;
    w.u[NX - 2][j] = 0;
  }

  // fix boundaries for v
  for (uint64_t i = 0; i < NX; i++) {
    w.v[i][0] = 0;
    w.v[i][NY - 1] = 0;
    w.v[i][NY - 2] = 0;
  }

  // updates elevation map
  for (uint64_t j = 1; j < NY; j++) {
    for (uint64_t i = 1; i < NX; i++) {
      w.e[i][j] -=
          dt / dx * ((w.u[i][j] - w.u[i - 1][j]) + (w.v[i][j] - w.v[i][j - 1]));
    }
  }
}

void to_file(const std::vector<grid_t> &water_history,
             const std::string &filename) {
  std::ofstream file(filename);
  file.write((const char *)(water_history.data()),
             sizeof(grid_t) * water_history.size());
}

void simulate(const Sim_Configuration config) {
  Water water_world = Water();

  std::vector<grid_t> water_history;
  auto begin = std::chrono::steady_clock::now();

  for (int t = 0; t < config.iter; ++t) {

    integrate(water_world, config.dt, config.dx, config.dy,
              config.g);

    if (t % config.data_period == 0) {
      water_history.push_back(water_world.e);
    }
  }

  auto end = std::chrono::steady_clock::now();

  to_file(water_history, config.filename);

  // std::cout << "checksum: " << std::accumulate(water_world.e.front().begin(),
  // water_world.e.back().end(), 0.0) << "\n";
  std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0
            << " sec"
            << "\n";
}

int main(int argc, char **argv) {
  auto config = Sim_Configuration({argv, argv + argc});
  // std::cout << "n_gangs:" << config.n_gangs << "\n";
  std::cout << NX << " x " << NY << "\n";
  std::cout << config.iter << "\n";

  simulate(config);
  return 0;
}
