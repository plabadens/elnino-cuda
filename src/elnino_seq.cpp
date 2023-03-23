#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <cassert>
#include <array>
#include <algorithm>
#include <math.h>
#include <cmath>



using real_t = float;
constexpr size_t NX = 300, NY = 400; // number of grid points
constexpr size_t XX = 3000000, YY = 4000000; // size of the grid 3000km x 4000km

real_t dx = XX/NX;     // Integration step size in the horizontal direction
real_t dy = YY/NY;
using grid_t = std::array<std::array<real_t, NY>, NX>;
double dt = 100;       // Size of the integration time step
real_t checksum = 0;
double beta = 2e-11;


class Sim_Configuration {
public:
    int iter = 1000000;  // Number of iterations
    real_t dt = 100;    // Size of the integration time step
    real_t g = 0.01;    // Gravitational acceleration
    real_t dx = XX/NX;  // Integration step size in the horizontal direction
    real_t dy = YY/NY;
    int data_period = 1000;  // how often to save coordinate to file
    std::string filename = "sw_output.data";   // name of the output file with history
    real_t tau_over_D = 0.003;
    real_t tau = 0.01;
    real_t tau_prime = 0.075;
    
    

    Sim_Configuration(std::vector <std::string> argument){
        for (long unsigned int i = 1; i<argument.size() ; i += 2){
            std::string arg = argument[i];
            if(arg=="-h"){ // Write help
                std::cout << "./par --iter <number of iterations> --dt <time step>"
                          << " --g <gravitational const> --dx <x grid size> --dy <y grid size>"
                          << "--fperiod <iterations between each save> --out <name of output file>\n";
                exit(0);
            } else if (i == argument.size() - 1)
                throw std::invalid_argument("The last argument (" + arg +") must have a value");
            else if(arg=="--iter"){
                if ((iter = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("iter most be a positive integer (e.g. -iter 1000)");
            } else if(arg=="--dt"){
                if ((dt = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dt most be a positive real number (e.g. -dt 0.05)");
            } else if(arg=="--g"){
                g = std::stod(argument[i+1]);
            } else if(arg=="--dx"){
                if ((dx = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dx most be a positive real number (e.g. -dx 1)");
            } else if(arg=="--dy"){
                if ((dy = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dy most be a positive real number (e.g. -dy 1)");
            } else if(arg=="--fperiod"){
                if ((data_period = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("fperiod most be a positive integer (e.g. -fperiod 100)");
            } /*else if(arg=="--n_gangs"){
                if ((n_gangs = std::stoi(argument[i+1])) < 0)
                    throw std::invalid_argument("n_gangs most be a positive integer (e.g. -n_gangs 14)");} */
             else if(arg=="--out"){
                filename = argument[i+1];
            }
            else{
                std::cout << "---> error: the argument type is not recognized \n";
            }
        }
    }
    
};

class Water {
public:
    grid_t u{}; // The speed in the horizontal direction.
    grid_t v{}; // The speed in the vertical direction.
    grid_t e{}; // The water elevation.
    real_t d_e_mean;
    real_t l_e_mean = 0;
    real_t r_e_mean = 0;
    int mean_update = 100;
    real_t epsilon = 0.75;

   
    
    std::array<double,NY+2> A; 
    std::array<double,NY+2> B; 
    

    
    double beta = 2e-11;
    Water() {
        //#pragma omp smd collapse(2)
        for (size_t i = 1; i < NX - 1; ++i) { 
            for (size_t j = 1; j < NY - 1; ++j) {
                real_t ii = 100.0 * (i - (NX - 100.0) / 2.0) / NX;
                real_t jj = 100.0 * (j - (NY - 2.0) / 2.0) / NY;
                e[i][j] = std::exp(-0.02 * (ii * ii + jj * jj));  // centered gaussian doesn't move w/ corriolis
                checksum += e[i][j]; 
                
            }
        }
        std::cout << checksum << "\n";
        

        // calculates f, scaling with grid size. Equator is off by one
        for (uint64_t i = 0; i < A.size(); i++) { 
            A[i] = beta * (double((NY) / 2) - double(i)) * dy * dt; //*dt *1000;
            B[i] = 0.25 * pow(A[i],2);
        }
        for (size_t i = 0; i < NX/8; ++i) { 
            for (size_t j = NY/8; j < 7*NY/8; ++j) {
                l_e_mean += e[i][j];
            }
        }
        
        for (size_t i = 7*NX/8; i < NX-1; ++i) { 
            for (size_t j = NY/8; j < 7*NY/8; ++j) {
                r_e_mean += e[i][j] ;
            }
        }
        d_e_mean = (l_e_mean-r_e_mean) / (NY*NX/4);
        
    }
};


void integrate(Water &w, const real_t tau_over_D, const real_t dt, const real_t dx, const real_t dy, const real_t g, int t,
              const real_t tau, const real_t tau_prime, int data_period) {

    //#pragma acc parallel present(w, NY, NX)  num_gangs(n_gangs)
    {
    
    //exchange_horizontal_ghost_lines(w.e);
    //#pragma acc loop
    for (uint64_t j = 0; j < NY; ++j) {
        w.e[0][j]      = w.e[NX-2][j]; 
        w.e[NX-1][j]   = w.e[1][j];
    }
    
    //exchange_horizontal_ghost_lines(w.v);
    //#pragma acc loop
    for (uint64_t j = 0; j < NY; ++j) {
        w.v[0][j]      = w.v[NX-2][j]; 
        w.v[NX-1][j]   = w.v[1][j];
    }
    
    //exchange_vertical_ghost_lines(w.e);
    //#pragma acc loop
    for (uint64_t i = 0; i < NX; ++i) {
        w.e[i][0] = w.e[i][NY-2];
        w.e[i][NY-1] = w.e[i][1];
    }
    
    
    //exchange_vertical_ghost_lines(w.u);
    //#pragma acc loop
        for (uint64_t i = 0; i < NX; ++i) {
        w.u[i][0] = w.u[i][NY-2];
        w.u[i][NY-1] = w.u[i][1];
    }
    }
    
    
    
    //#pragma acc parallel loop gang present(w, NY, NX)  collapse(2) num_gangs(n_gangs)
    for (uint64_t i = 0; i<NX; i++) {
        for (uint64_t j = 0; j < NY; j++) {
            
            real_t wu_copy = w.u[i][j];
            real_t wv_copy = w.v[i][j];
            
            
            real_t u_star = dt * (g/dx * (w.e[i+1][j] - w.e[i][j]) + (tau - tau_prime * w.d_e_mean )/ 1000 / 100);
            real_t v_star = dt * (g/dy * (w.e[i][j+1] - w.e[i][j]));
           
            w.u[i][j] = w.u[i][j] * (1-beta) - (u_star - w.B[j] * wu_copy + w.A[j] * wv_copy) / (1+w.B[j]);
            w.v[i][j] = w.v[i][j] * (1-beta) - (v_star - w.B[j] * wv_copy - w.A[j] * wu_copy) / (1+w.B[j]);
            

        }
    }
    
    // fix boundaries for u
    for (uint64_t j = 0; j<NY; j++) {
        w.u[0][j] = 0;
        w.u[NX-1][j] = 0;
        w.u[NX-2][j] = 0;
    }
    
    // fix boundaries for v
    for (uint64_t i = 0; i<NX; i++) {
        w.v[i][0] = 0;
        w.v[i][NY-1] = 0;
        w.v[i][NY-2] = 0;

    }
    
    // updates elevation map
    //#pragma acc parallel loop gang present(w, NY, NX)  collapse(2) num_gangs(n_gangs)
    for (uint64_t j = 1; j < NY; j++) {
        for (uint64_t i = 1; i< NX; i++) {
            w.e[i][j] -= dt * ((w.u[i][j] - w.u[i-1][j])/dx + (w.v[i][j] - w.v[i][j-1])/dy);
            checksum += w.e[i][j];
            
        }
    }

    // filter elevation map with epsilon

    /*
    for (uint64_t j = 1; j < NY; j++) {
        for (uint64_t i = 1; i< NX; i++) {
            w.e[i][j] = w.e[i][j] * (1-w.epsilon) + w.epsilon * 0.25 * (w.e[i][j-1] + w.e[i][j+1] + w.e[i-1][j] + w.e[i+1][j]);
        }
    }
    */

    // update the mean elevation difference
    if (t % 5000 == 0) {
        // wipe the history 
        w.l_e_mean = 0;
        w.r_e_mean = 0;
        for (size_t i = 0; i < NX/2; ++i) { 
            for (size_t j = 7*NY/16; j < 9*NY/16; ++j) {
                w.l_e_mean += w.e[i][j];
            }
        }

        for (size_t i = 1*NX/2; i < NX-1; ++i) { 
            for (size_t j = 7*NY/16; j < 9*NY/16; ++j) {
                w.r_e_mean += w.e[i][j];
            }
        }
        w.d_e_mean = (w.l_e_mean-w.r_e_mean) / (NY*NX/8);
        //if (t % data_period == 0) {
        //    std::cout << w.d_e_mean << " , " <<  (tau - tau_prime * w.d_e_mean )<< "\n";
        //}
        
    }
    

}

void to_file(const std::vector<grid_t> &water_history, const std::string &filename){
    std::ofstream file(filename);
    file.write((const char*)(water_history.data()), sizeof(grid_t)*water_history.size());
}


void simulate(const Sim_Configuration config) {
    Water water_world = Water();

    std::vector <grid_t> water_history;
    auto begin = std::chrono::steady_clock::now();
    
    //#pragma acc data copyin(water_world, NX, NY) copyout(water_world.e)
    
    for (int t = 0; t < config.iter; ++t) {
        
        integrate(water_world, config.tau_over_D, config.dt, config.dx, config.dy, config.g, t, config.tau, config.tau_prime, config.data_period);
        
        if (t % config.data_period == 0) {
            //#pragma acc update self(water_world.e)
            
            water_history.push_back(water_world.e);
        }
        
        
    }
    
    auto end = std::chrono::steady_clock::now();
    
    to_file(water_history, config.filename);

    // std::cout << "checksum: " << std::accumulate(water_world.e.front().begin(), water_world.e.back().end(), 0.0) << "\n";
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << "\n";
    std::cout << "checksum: " << checksum/config.iter << "\n";
}

int main(int argc, char **argv) {
    auto config = Sim_Configuration({argv, argv+argc});
    // std::cout << "n_gangs:" << config.n_gangs << "\n";
    std::cout << NX << " x " << NY << "\n";
    std::cout << config.iter <<  "\n";
    
    simulate(config);
    return 0;
}
