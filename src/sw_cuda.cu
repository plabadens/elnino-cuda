#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda/std/array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>

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
constexpr size_t NX = 512, NY = 512; // World Size
// Threads per block for the GPU: 16*16 = 256 threads
constexpr size_t THREADX = 16, THREADY = 16;
using grid_t = cuda::std::array<cuda::std::array<real_t, NX>, NY>;

// Declare constants in device memory
__constant__ real_t DT;
__constant__ real_t DX;
__constant__ real_t DY;
__constant__ real_t G;

class Sim_Configuration {
public:
  int iter = 100;        // Number of iterations
  real_t dt = 0.05;      // Size of the integration time step
  real_t g = 9.80665;    // Gravitational acceleration
  real_t dx = 1;         // Integration step size in the horizontal direction
  real_t dy = 1;         // Integration step size in the vertical direction
  int data_period = 100; // how often to save coordinate to file
  std::string filename =
      "sw_output.data"; // name of the output file with history

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
              "dy most be a positive integer (e.g. -fperiod 100)");
      } else if (arg == "--out") {
        filename = argument[i + 1];
      } else {
        std::cout << "---> error: the argument type is not recognized \n";
      }
    }
  }
};

/** Representation of a water world including ghost lines, which is a "1-cell
 * padding" of rows and columns around the world. These ghost lines is a
 * technique to implement periodic boundary conditions. */
struct water {
  grid_t u; // The speed in the horizontal direction.
  grid_t v; // The speed in the vertical direction.
  grid_t e; // The water elevation.
};

/** Initialize the elevation map of a given water world
 *
 * @param elevation   Reference to an array of the elevation map
 */
__global__ void initialize_water_elevation(grid_t &elevation) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  real_t ii = 100.0 * (i - (NY - 2.0) / 2.0) / NY;
  real_t jj = 100.0 * (j - (NX - 2.0) / 2.0) / NX;
  elevation[i][j] = expf(-0.02 * (ii * ii + jj * jj));
}

/* Write a history of the water heights to an ASCII file
 *
 * @param water_history  Vector of the all water worlds to write
 * @param filename       The output filename of the ASCII file
 */
void to_file(const std::vector<grid_t> &water_history,
             const std::string &filename) {
  std::ofstream file(filename);
  file.write((const char *)(water_history.data()),
             sizeof(grid_t) * water_history.size());
}

/** Exchange the horizontal ghost lines i.e. copy the second data row to the
 * very last data row and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the
 * speed in the horizontal direction `u`.
 * @param shape  The shape of data including the ghost lines.
 */
__device__ void exchange_horizontal_ghost_lines(grid_t &data, int &i, int &j) {
  if (i == 0) {
    data[0][j] = data[NY - 2][j];
    data[NY - 1][j] = data[1][j];
  }
}

/** Exchange the vertical ghost lines i.e. copy the second data column to the
 * rightmost data column and vice versa.
 *
 * @param data   The data update, which could be the water elevation `e` or the
 * speed in the vertical direction `v`.
 * @param shape  The shape of data including the ghost lines.
 */
__device__ void exchange_vertical_ghost_lines(grid_t &data, int &i, int &j) {
  if (j == 0) {
    data[i][0] = data[i][NX - 2];
    data[i][NX - 1] = data[i][1];
  }
}

__device__ void integrate_uv(water &w, int &i, int &j) {
  if (i + 1 < NY && j + 1 < NX) {
    w.u[i][j] -= DT / DX * G * (w.e[i][j + 1] - w.e[i][j]);
    w.v[i][j] -= DT / DY * G * (w.e[i + 1][j] - w.e[i][j]);
  }
}

__device__ void integrate_e(water &w, int &i, int &j) {
  if (i - 1 >= 0 && j - 1 >= 0) {
    w.e[i][j] -= DT / DX * (w.u[i][j] - w.u[i][j - 1]) +
                 DT / DY * (w.v[i][j] - w.v[i - 1][j]);
  }
}

/** One integration step
 *
 * @param w The water world to update.
 */
__global__ void integrate(water &w) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  exchange_horizontal_ghost_lines(w.e, i, j);
  exchange_horizontal_ghost_lines(w.v, i, j);
  exchange_vertical_ghost_lines(w.u, i, j);
  // wait for other threads to finish writing e
  __threadfence();
  exchange_vertical_ghost_lines(w.e, i, j);
  __threadfence();

  integrate_uv(w, i, j);
  // wait for other threads to finish writing u, v
  __threadfence();
  integrate_e(w, i, j);
}

/** Simulation of shallow water
 *
 * @param num_of_iterations  The number of time steps to simulate
 * @param size               The size of the water world excluding ghost lines
 * @param output_filename    The filename of the written water world history
 * (HDF5 file)
 */
void simulate(const Sim_Configuration config) {
  // Copy all constants to special GPU constants memory
  checkCuda(cudaMemcpyToSymbol(DT, &config.dt, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(DX, &config.dx, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(DY, &config.dy, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbol(G, &config.g, sizeof(real_t)));

  // Allocate the water world memory shared between GPU and CPU
  water *water_world;
  checkCuda(cudaMallocManaged((void **)&water_world, sizeof(water)));

  // Calculate the dimentions of the 2D GPU compute grid
  dim3 threadsPerBlock(THREADY, THREADX);
  dim3 numBlocks(NY / threadsPerBlock.y, NX / threadsPerBlock.x);

  initialize_water_elevation<<<numBlocks, threadsPerBlock>>>(water_world->e);

  std::vector<grid_t> water_history;
  auto begin = std::chrono::steady_clock::now();

  for (int t = 0; t < config.iter; ++t) {
    integrate<<<numBlocks, threadsPerBlock>>>(*water_world);
    /*if (t % config.data_period == 0) {
      checkCuda(cudaDeviceSynchronize());
      water_history.push_back(water_world->e);
    }*/
  }

  checkCuda(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();

  to_file(water_history, config.filename);
  std::cout << "checksum: "
            << std::accumulate(water_world->e.front().begin(),
                               water_world->e.back().end(), 0.0)
            << std::endl;
  std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0
            << " sec" << std::endl;

  checkCuda(cudaFree(water_world));
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
  auto config = Sim_Configuration({argv, argv + argc});
  simulate(config);
  return 0;
}
