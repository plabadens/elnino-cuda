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
constexpr size_t XX = 3000000, YY = 4000000; // size of the grid 3000km x 4000km
constexpr size_t THREADX = 16, THREADY = 16;
#define ELEMENT_PTR(base, pitch, row, col)                                     \
  ((real_t *)((char *)base + (row) * (pitch) + (col) * sizeof(real_t)))

// Device constants
__constant__ int NX, NY;
__constant__ real_t DT, DX, DY;
__constant__ real_t G;
__constant__ real_t BETA;
__constant__ real_t EPSILON;
__constant__ real_t TAU;
__constant__ real_t TAU_PRIME;

class Sim_Configuration {
public:
  int iter = 100000; // Number of iterations
  int nx = 300;
  int ny = 400;
  int grid_dim;
  real_t dt = 100;     // Size of the integration time step
  real_t g = 0.01;     // Gravitational acceleration
  real_t dx = XX / nx; // Integration step size in the horizontal direction
  real_t dy = YY / ny;
  real_t beta = 2e-11;
  real_t epsilon = 0;
  int data_period = 1000; // how often to save coordinate to file
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
      } else if (arg == "--nx") {
        if ((nx = std::stod(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "nx most be a positive real number (e.g. --nx 300)");
      } else if (arg == "--ny") {
        if ((ny = std::stod(argument[i + 1])) < 0)
          throw std::invalid_argument(
              "ny most be a positive real number (e.g. --ny 400)");
      } else if (arg == "--eps") {
        epsilon = std::stod(argument[i + 1]);
      } else if (arg == "--g") {
        g = std::stod(argument[i + 1]);
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
    grid_dim = nx * ny;
  }
};

struct means {
  real_t mean;
  real_t mean_left;
  real_t mean_right;
};

__global__ void update_elevation_mean(real_t *eta, size_t pitch, means &em) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {

      real_t e_t = *ELEMENT_PTR(eta, pitch, j, i);

      if (i < NX / 8 && j >= 7 * NY / 16 && j < 9 * NY / 16) {
        atomicAdd(&em.mean_left, e_t);
      } else if (i >= 7 * NX / 8 && i < NX && j >= 7 * NY / 16 &&
                 j < 9 * NY / 16) {
        atomicAdd(&em.mean_right, e_t);
      }
    }
  __threadfence();

  if (blockIdx.x * blockDim.x + threadIdx.x == 0 &&
      blockIdx.y * blockDim.y + threadIdx.y == 0) {
    em.mean_left = em.mean_left / (real_t)(NY * NX / 16.0);
    em.mean_right = em.mean_right / (real_t)(NY * NX / 16.0);

    em.mean = em.mean_left - em.mean_right;

    em.mean_left = 0;
    em.mean_right = 0;
  }
}

__global__ void initialize_water(real_t *eta, real_t *u, real_t *v,
                                 size_t pitch, real_t *A, real_t *B,
                                 means &em) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i > 0 && i < NX - 1 && j > 0 && i < NY - 1) {
        real_t ii = 100.0 * (i - (NX + 250.0) / 2.0) / NX;
        real_t jj = 100.0 * (j - (NY - 2.0) / 2.0) / NY;

        *ELEMENT_PTR(eta, pitch, j, i) = expf(-0.02 * (ii * ii + jj * jj)) * 10;
        *ELEMENT_PTR(u, pitch, j, i) = 0;
        *ELEMENT_PTR(v, pitch, j, i) = 0;
      }

      if (i == 0) {
        A[j] = BETA * ((real_t)(NY / 2) - (real_t)(j)) * DY * DT;
        B[j] = 0.25 * pow(A[j], 2);

        if (j == 0) {
          em.mean_left = 0;
          em.mean_right = 0;
        }
      }
    }
};

__global__ void integrate_velocity(real_t *eta, real_t *u, real_t *v,
                                   size_t pitch, real_t *A, real_t *B,
                                   means &em, int t, int t_wind_stop) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i + 1 < NX && j + 1 < NY) {
        real_t *u_t = ELEMENT_PTR(u, pitch, j, i);
        real_t *v_t = ELEMENT_PTR(v, pitch, j, i);

        real_t *e_m_t = ELEMENT_PTR(eta, pitch, j, i);
        real_t *e_d_t = ELEMENT_PTR(eta, pitch, j + 1, i);
        real_t *e_r_t = ELEMENT_PTR(eta, pitch, j, i + 1);

        real_t u_star;
        if (t < t_wind_stop) {
          u_star = DT * (G / DX * (*e_r_t - *e_m_t) +
                         (TAU - TAU_PRIME * em.mean) / 1000 / 100);
        } else {
          u_star = DT * (G / DX * (*e_r_t - *e_m_t));
        }

        real_t v_star = DT * (G / DY * (*e_d_t - *e_m_t));

        *u_t = *u_t * (1 - BETA) -
               (u_star - B[j] * (*u_t) + A[j] * (*v_t)) / (1 + B[j]);
        *v_t = *v_t * (1 - BETA) -
               (v_star - B[j] * (*v_t) - A[j] * (*u_t)) / (1 + B[j]);
      }

      if (i == 0) {
        *ELEMENT_PTR(u, pitch, j, 0) = 0;
        *ELEMENT_PTR(u, pitch, j, NX - 1) = 0;
        *ELEMENT_PTR(u, pitch, j, NX - 2) = 0;
      }

      if (j == 0) {
        *ELEMENT_PTR(v, pitch, 0, i) = 0;
        *ELEMENT_PTR(v, pitch, NY - 1, i) = 0;
        *ELEMENT_PTR(v, pitch, NY - 2, i) = 0;
      }
    }
}

__global__ void integrate_elevation(real_t *eta, real_t *u, real_t *v,
                                    size_t pitch) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y) {
      if (i > 0 && j > 0) {
        real_t *e_m = ELEMENT_PTR(eta, pitch, j, i);
        real_t *e_r = ELEMENT_PTR(eta, pitch, j, i + 1);
        real_t *e_d = ELEMENT_PTR(eta, pitch, j + 1, i);

        real_t *u_m = ELEMENT_PTR(u, pitch, j, i);
        real_t *u_l = ELEMENT_PTR(u, pitch, j, i - 1);

        real_t *v_m = ELEMENT_PTR(v, pitch, j, i);
        real_t *v_u = ELEMENT_PTR(v, pitch, j - 1, i);

        *e_m -= DT * ((*u_m - *u_l) / DX * (*e_m + 100) +
                      (*v_m - *v_u) / DY * (*e_m + 100));

        if ((*e_r - *e_m) > 1e-15) {
          *e_m -= DT * (*u_m) * (*e_r - *e_m) / DX;
        }

        if ((*e_d - *e_m) > 1e-15) {
          *e_m -= DT * (*v_m) * (*e_d - *e_m) / DY;
        }
      }
    }
}

__global__ void shapiro_filter(real_t *input, real_t *output, size_t pitch) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NX;
       i += blockDim.x * gridDim.x)
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < NY;
         j += blockDim.y * gridDim.y)
      if (i < NX && j < NY) {
        real_t sum = 0.0;
        for (int k = -1; k <= 1; k++) {
          for (int l = -1; l <= 1; l++) {
            int x = i + k;
            int y = j + l;
            if (x >= 0 && x < NX && y >= 0 && y < NY) {
              sum += *ELEMENT_PTR(input, pitch, y, x);
            }
          }
        }
        real_t *input_cell = ELEMENT_PTR(input, pitch, j, i);
        real_t *output_cell = ELEMENT_PTR(output, pitch, j, i);

        *output_cell = (1 - EPSILON) * (*input_cell) + EPSILON * sum / 9;
      }
}

void to_file(const real_t *water_history, const Sim_Configuration config) {
  std::ofstream file(config.filename);
  file.write((const char *)(water_history),
             sizeof(real_t) * config.grid_dim * config.data_iter);
}

void print_checksum(real_t *elevation, const char *label,
                    const Sim_Configuration config) {
  real_t sum = 0;

  for (int i = 0; i < config.nx; i++)
    for (int j = 0; j < config.ny; j++)
      sum += elevation[j * config.nx + i];

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
  checkCuda(cudaMemcpyToSymbolAsync(NX, &config.nx, sizeof(uint)));
  checkCuda(cudaMemcpyToSymbolAsync(NY, &config.ny, sizeof(uint)));
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
  real_t *h_water_history;
  checkCuda(
      cudaMallocHost(&h_water_history, config.grid_dim * config.data_iter * sizeof(real_t)));
  real_t *d_eta, *d_eta_copy, *d_u, *d_v;
  size_t pitch;
  checkCuda(
      cudaMallocPitch(&d_eta, &pitch, config.nx * sizeof(real_t), config.ny));
  checkCuda(cudaMallocPitch(&d_eta_copy, &pitch, config.nx * sizeof(real_t),
                            config.ny));
  checkCuda(
      cudaMallocPitch(&d_u, &pitch, config.nx * sizeof(real_t), config.ny));
  checkCuda(
      cudaMallocPitch(&d_v, &pitch, config.nx * sizeof(real_t), config.ny));
  real_t *d_A, *d_B;
  checkCuda(cudaMalloc(&d_A, config.ny * sizeof(real_t)));
  checkCuda(cudaMalloc(&d_B, config.ny * sizeof(real_t)));
  means *d_eta_means;
  checkCuda(cudaMalloc(&d_eta_means, sizeof(means)));

  // Calculate the dimentions of the 2D GPU compute grid
  dim3 threadsPerBlock(THREADX, THREADY);
  dim3 numBlocks(config.nx / threadsPerBlock.x, config.ny / threadsPerBlock.y);

  initialize_water<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_eta, d_u, d_v, pitch, d_A, d_B, *d_eta_means);

  checkCuda(cudaEventRecord(start, stream));

  for (int t = 0; t < config.iter; t++) {
    integrate_velocity<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_eta, d_u, d_v, pitch, d_A, d_B, *d_eta_means, t, config.wind_stop);
    integrate_elevation<<<numBlocks, threadsPerBlock, 0, stream>>>(d_eta, d_u,
                                                                   d_v, pitch);

    checkCuda(cudaMemcpy2DAsync(d_eta_copy, pitch, d_eta, pitch,
                                config.nx * sizeof(real_t), config.ny,
                                cudaMemcpyDeviceToDevice, stream));
    shapiro_filter<<<numBlocks, threadsPerBlock, 0, stream>>>(d_eta_copy, d_eta,
                                                              pitch);
      
    
    if (t % config.data_period == 0) {
      int i = t / config.data_period;
      printf("\rdata period: %3d / %3d", i + 1, config.data_iter);
      fflush(stdout);

      update_elevation_mean<<<numBlocks, threadsPerBlock, 0, stream>>>(
          d_eta, pitch, *d_eta_means);
      checkCuda(cudaMemcpy2DAsync(&h_water_history[config.grid_dim * i],
                                  config.nx * sizeof(real_t), d_eta, pitch,
                                  config.nx * sizeof(real_t), config.ny,
                                  cudaMemcpyDeviceToHost, stream));
    }
  }
  printf("\n");

  checkCuda(cudaEventRecord(stop, stream));

  checkCuda(cudaFreeAsync(d_eta, stream));
  checkCuda(cudaFreeAsync(d_eta_copy, stream));
  checkCuda(cudaFreeAsync(d_u, stream));
  checkCuda(cudaFreeAsync(d_v, stream));
  checkCuda(cudaFreeAsync(d_A, stream));
  checkCuda(cudaFreeAsync(d_B, stream));
  checkCuda(cudaFreeAsync(d_eta_means, stream));
  checkCuda(cudaEventSynchronize(stop));

  float time_ms = 0;
  checkCuda(cudaEventElapsedTime(&time_ms, start, stop));

  checkCuda(cudaStreamDestroy(stream));

  print_checksum(&h_water_history[0], "first", config);
  print_checksum(&h_water_history[config.grid_dim * (config.data_iter - 1)], "last", config);

  to_file(h_water_history, config);
  std::cout << "elapsed time: " << time_ms / 1000 << " sec"
            << "\n";

  checkCuda(cudaFreeHost(h_water_history));
}

int main(int argc, char **argv) {
  auto config = Sim_Configuration({argv, argv + argc});

  print_devices();

  std::cout << "dimensions: " << config.nx << "x" << config.ny << "\n";
  std::cout << "iterations: " << config.iter << "\n";

  launch(config);
  return 0;
}
