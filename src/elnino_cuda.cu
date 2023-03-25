#include <cuda/std/array>
#include <fstream>
#include <iostream>
#include <vector>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#ifndef NDEBUG
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
__constant__ real_t EPSILON;
__constant__ real_t TAU;
__constant__ real_t TAU_PRIME;

class Sim_Configuration {
public:
  uint iter = 500000;   // Number of iterations
  real_t dt = 100;     // Size of the integration time step
  real_t g = 0.01;     // Gravitational acceleration
  real_t dx = XX / NX; // Integration step size in the horizontal direction
  real_t dy = YY / NY;
  real_t beta = 2e-11;
  real_t epsilon = 0.001;
  uint data_period = 1000; // how often to save coordinate to file
  uint data_iter;
  uint wind_stop = 20000;
  std::string filename =
      "sw_output.data"; // name of the output file with history
  real_t tau = 0.01;
  real_t tau_prime = 0.0075;

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

    data_iter = iter / data_period;
  }
};

struct water {
  grid_t u;
  grid_t v;
  grid_t e;
  column_t A;
  column_t B;
  real_t e_mean;
  real_t e_mean_left;
  real_t e_mean_right;
};

__global__ void update_elevation_mean(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i == 0 && j == 0) {
        w.e_mean_left = 0;
        w.e_mean_right = 0;
      }
      __threadfence();

      if (i < NX / 2) {
        atomicAdd_system(&w.e_mean_left, w.e[i][j]);
      } else {
        atomicAdd_system(&w.e_mean_right, w.e[i][j]);
      }
      __threadfence();

      if (i == 0 && j == 0) {
        w.e_mean = (w.e_mean_left - w.e_mean_right) / (real_t)(NY * NX / 8);
      }
    }
}

__global__ void initialize_water(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i > 0 && i < NX - 1 && j > 0 && i < NY - 1) {
        real_t ii = 100.0 * (i - (NX - 100.0) / 2.0) / NX;
        real_t jj = 100.0 * (j - (NY - 2.0) / 2.0) / NY;

        w.e[i][j] = expf(-0.02 * (ii * ii + jj * jj)) * 4;
        w.u[i][j] = 0;
        w.v[i][j] = 0;
      }

      if (i == 0) {
        w.A[j] = BETA * ((real_t)(NY / 2) - (real_t)(j)) * DY * DT;
        w.B[j] = 0.25 * pow(w.A[j], 2);
      }
    }
};

__global__ void integrate_velocity(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y)
      if (i + 1 < NX && j + 1 < NY) {
        real_t u_t = w.u[i][j];
        real_t u_star = DT * (G / DX * (w.e[i + 1][j] - w.e[i][j]));

        real_t v_t = w.v[i][j];
        real_t v_star = DT * (G / DY * (w.e[i][j + 1] - w.e[i][j]));

        w.u[i][j] = w.u[i][j] * (1 - BETA) -
                    (u_star - w.B[j] * u_t + w.A[j] * v_t) / (1 + w.B[j]);
        w.v[i][j] = w.v[i][j] * (1 - BETA) -
                    (v_star - w.B[j] * v_t - w.A[j] * u_t) / (1 + w.B[j]);
      }
}

__global__ void integrate_elevation(water &w) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
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
      __threadfence();

      if (i > 0 && j > 0) {
        real_t e_mid_t = w.e[i][j];
        real_t e_right_t = w.e[i + 1][j];
        real_t e_up_t = w.e[i][j + 1];
        w.e[i][j] -= DT * ((w.u[i][j] - w.u[i - 1][j]) / DX * (e_mid_t + 100) +
                           (w.v[i][j] - w.v[i][j - 1]) / DY * (e_mid_t + 100));

        if ((e_right_t - e_mid_t) > 1e-15) {
          w.e[i][j] -= DT * w.u[i][j] * (e_right_t - e_mid_t) / DX;
        }

        if ((e_up_t - e_mid_t) > 1e-15) {
          w.e[i][j] -= DT * w.v[i][j] * (e_up_t - e_mid_t) / DY;
        }
      }
    }
}

__global__ void shapiro_filter(water &w, grid_t &e_t) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y)
      if (i > 0 && i + 1 < NX && j > 0 && j + 1 < NY) {
        w.e[i][j] =
            e_t[i][j] * (1 - EPSILON) +
            EPSILON * 0.25 *
                (e_t[i][j - 1] + e_t[i][j + 1] + e_t[i - 1][j] + e_t[i + 1][j]);
      }
}

void to_file(const grid_t *water_history, const Sim_Configuration config) {
  std::ofstream file(config.filename);
  file.write((const char *)(water_history), sizeof(grid_t) * config.data_iter);
}

void print_checksum(grid_t &elevation, const char *label) {
  real_t sum = 0;

  for (size_t i = 0; i < NX; i++)
    for (size_t j = 0; j < NY; j++)
      sum += elevation[i][j];

  printf("checksum [%s]: %f \n", label, sum);
}

void print_devices() {
  int nDevices;

  printf("devices:\n");

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("  %d: %s: %d.%d\n", i, prop.name, prop.major, prop.minor);
  }
}

void launch(const Sim_Configuration config) {
  // Initialize a stream for asynchronous operations
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));

  // Setup timing using CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Copy constant to special symbols
  checkCuda(cudaHostRegister((void **)&config, sizeof(config),
                             cudaHostRegisterReadOnly));
  checkCuda(cudaMemcpyToSymbolAsync(DT, &config.dt, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(DX, &config.dx, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(DY, &config.dy, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(G, &config.g, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(BETA, &config.beta, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(EPSILON, &config.epsilon, sizeof(real_t)));
  checkCuda(cudaMemcpyToSymbolAsync(TAU, &config.tau, sizeof(real_t)));
  checkCuda(
      cudaMemcpyToSymbolAsync(TAU_PRIME, &config.tau_prime, sizeof(real_t)));

  // Allocate the water world memory on both the host and GPU
  grid_t *h_water_history;
  checkCuda(
      cudaMallocHost(&h_water_history, sizeof(grid_t) * config.data_iter));
  water *d_water_world;
  checkCuda(cudaMallocAsync(&d_water_world, sizeof(water), stream));
  grid_t *d_tmp_eta;
  checkCuda(cudaMallocAsync(&d_tmp_eta, sizeof(grid_t), stream));

  // Calculate the dimentions of the 2D GPU compute grid
  dim3 threadsPerBlock(THREADX, THREADY);
  dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);

  initialize_water<<<numBlocks, threadsPerBlock, 0, stream>>>(*d_water_world);

  checkCuda(cudaEventRecord(start, stream));

  for (int t = 0; t < config.iter; ++t) {

    integrate_velocity<<<numBlocks, threadsPerBlock, 0, stream>>>(
        *d_water_world);
    integrate_elevation<<<numBlocks, threadsPerBlock, 0, stream>>>(
        *d_water_world);

    checkCuda(cudaMemcpyAsync(d_tmp_eta, &d_water_world->e, sizeof(grid_t),
                              cudaMemcpyDeviceToDevice));
    shapiro_filter<<<numBlocks, threadsPerBlock, 0, stream>>>(*d_water_world,
                                                              *d_tmp_eta);

    if (t % config.data_period == 0) {
      int i = t / config.data_period;
      printf("\rdata period: %3d / %3d", i + 1, config.data_iter);
      fflush(stdout);

      update_elevation_mean<<<numBlocks, threadsPerBlock, 0, stream>>>(
          *d_water_world);
      checkCuda(cudaMemcpyAsync(&h_water_history[i], &d_water_world->e,
                                sizeof(grid_t), cudaMemcpyDeviceToHost,
                                stream));
    }
  }
  printf("\n");

  checkCuda(cudaEventRecord(stop, stream));

  checkCuda(cudaFreeAsync(d_water_world, stream));
  checkCuda(cudaFreeAsync(d_tmp_eta, stream));
  checkCuda(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCuda(cudaEventElapsedTime(&time_ms, start, stop));

  checkCuda(cudaStreamDestroy(stream));

  print_checksum(h_water_history[0], "first");
  print_checksum(h_water_history[config.data_iter - 1], "last");

  to_file(h_water_history, config);
  std::cout << "elapsed time: " << time_ms / 1000 << " sec"
            << "\n";

  checkCuda(cudaFreeHost(h_water_history));
}

int main(int argc, char **argv) {
  auto config = Sim_Configuration({argv, argv + argc});

  print_devices();

  std::cout << "dimensions: " << NX << "x" << NY << "\n";
  std::cout << "iterations: " << config.iter << "\n";

  launch(config);
  return 0;
}
