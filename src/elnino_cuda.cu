#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda/std/array>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

using real_t = float;
constexpr int NX = 300, NY = 400;            // number of grid points
constexpr size_t XX = 3000000, YY = 4000000; // size of the grid 3000km x 4000km
constexpr size_t THREADX = 16, THREADY = 16;

real_t dx = XX / NX; // Integration step size in the horizontal direction
real_t dy = YY / NY;
using grid_t = cuda::std::array<cuda::std::array<real_t, NY>, NX>;
using column_t = cuda::std::array<real_t, NY>;

// Device constants
__constant__ real_t DT;
__constant__ real_t DX;
__constant__ real_t DY;
__constant__ real_t G;
__constant__ real_t BETA;

class Sim_Configuration {
public:
  int iter = 1000000;  // Number of iterations
  real_t dt = 100;     // Size of the integration time step
  real_t g = 0.01;     // Gravitational acceleration
  real_t dx = XX / NX; // Integration step size in the horizontal direction
  real_t dy = YY / NY;
  real_t beta = 2e-11;
  int data_period = 10000; // how often to save coordinate to file
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
      } else if (arg == "--out") {
        filename = argument[i + 1];
      } else {
        std::cout << "---> error: the argument type is not recognized \n";
      }
    }
  }
};

struct water {
  grid_t u;
  grid_t v;
  grid_t e;
  column_t f;
};

__global__ void initialize_water(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i > 0 && i < NX - 1 && j > 0 && i < NY - 1) {
        real_t ii = 100.0 * (i - (NX - 100.0) / 2.0) / NX;
        real_t jj = 100.0 * (j - (NY - 2.0) / 2.0) / NY;

        w.e[i][j] = expf(-0.02 * (ii * ii + jj * jj));
        w.u[i][j] = 0;
        w.v[i][j] = 0;
      }

      if (i == 0) {
        w.f[j] = BETA * ((real_t)(NY / 2) - (real_t)(j)) * DY;
      }
    }
};

__device__ void integrate_velocity(water &w, int &i, int &j) {
  if (i + 1 < NX && j + 1 < NY) {
    real_t u_t = w.u[i][j];

    w.u[i][j] -=
        DT * (G / DX * (w.e[i + 1][j] - w.e[i][j]) + w.f[j] * w.v[i][j]);
    w.v[i][j] -= DT * (G / DY * (w.e[i][j + 1] - w.e[i][j]) - w.f[j] * u_t);
  }
}

__device__ void set_boundary_conditions(water &w, int &i, int &j) {
  if (i == 0) {
    w.u[0][j] = 0;
    w.u[NX - 1][j] = 0;
    w.u[NX - 2][j] = 0;
  }

  if (j == 0) {
    w.v[i][0] = 0;
    w.v[i][NY - 1] = 0;
    w.v[i][NY - 2] = 0;
  }
}

__device__ void update_elevation(water &w, int &i, int &j) {
  if (i > 0 && j > 0) {
    w.e[i][j] -= DT * ((w.u[i][j] - w.u[i - 1][j]) / DX +
                       (w.v[i][j] - w.v[i][j - 1]) / DY);
  }
};

__global__ void integrate(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      integrate_velocity(w, i, j);
      set_boundary_conditions(w, i, j);
      update_elevation(w, i, j);
    }
}

void to_file(const std::vector<grid_t> &water_history,
             const std::string &filename) {
  std::ofstream file(filename);
  file.write((const char *)(water_history.data()),
             sizeof(grid_t) * water_history.size());
}

__host__ __device__ void print_checksum(grid_t &elevation) {
  real_t sum = 0;

  for (size_t i = 0; i < NX; i++)
    for (size_t j = 0; j < NY; j++)
      sum += elevation[i][j];

  printf("checksum: %f \n", sum);
}

void simulate(const Sim_Configuration config) {
  // Copy constant to special symbols
  checkCuda(cudaMemcpyToSymbol(DT, &config.dt, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(DX, &config.dx, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(DY, &config.dy, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(G, &config.g, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(BETA, &config.beta, sizeof(real_t)));

  // Allocate the water world memory on both the host and GPU
  grid_t *h_elevation = (grid_t *)malloc(sizeof(grid_t));
  water *d_water_world;
  checkCuda(cudaMalloc((void **)&d_water_world, sizeof(water)));

  // Calculate the dimentions of the 2D GPU compute grid
  dim3 threadsPerBlock(THREADX, THREADY);
  dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);

  initialize_water<<<numBlocks, threadsPerBlock>>>(*d_water_world);
  checkCuda(cudaMemcpy(h_elevation, &d_water_world->e, sizeof(grid_t),
                       cudaMemcpyDeviceToHost));
  print_checksum(*h_elevation);

  std::vector<grid_t> water_history;
  auto begin = std::chrono::steady_clock::now();

  for (int t = 0; t < config.iter; ++t) {

    integrate<<<numBlocks, threadsPerBlock>>>(*d_water_world);

    if (t % config.data_period == 0) {
      checkCuda(cudaMemcpy(h_elevation, &d_water_world->e, sizeof(grid_t),
                           cudaMemcpyDeviceToHost));
      water_history.push_back(*h_elevation);
    }
  }

  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaFree(d_water_world));
  auto end = std::chrono::steady_clock::now();

  to_file(water_history, config.filename);

  print_checksum(*h_elevation);
  std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0
            << " sec"
            << "\n";

  free(h_elevation);
}

int main(int argc, char **argv) {
  auto config = Sim_Configuration({argv, argv + argc});

  std::cout << "dimensions: " << NX << "x" << NY << "\n";
  std::cout << "iterations: " << config.iter << "\n";

  simulate(config);
  return 0;
}