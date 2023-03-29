{
  description = "Simple flake for C++ code";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }: 
    flake-utils.lib.eachDefaultSystem 
      (system: 
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in {
          packages.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              catch2_3
              gdb
              gnumake
              valgrind
              gcc
              ffmpeg
              cudaPackages.cudatoolkit
              cudaPackages.nsight_compute
              cmake
            ];

           shellHook = ''
              export CUDA_PATH=${pkgs.cudatoolkit}
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.ncurses5}/lib
              export EXTRA_LDFLAGS="-L/lib -L/run/opengl-driver/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
           '';
          };
          devShells.default = self.packages.${system}.default;
        }
    );
}
