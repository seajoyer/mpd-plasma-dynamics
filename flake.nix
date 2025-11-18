{
  description = "C++ development environment with OpenMP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Use the same LLVM version for consistency
        llvmPackages = pkgs.llvmPackages_17;  # or _16, _18 depending on your preference
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Use clang from the same LLVM package set
            llvmPackages.clang
            llvmPackages.openmp
            mpi
            
            # Build tools
            libgcc
            cmake
            gnumake
            pkg-config

            # Debugging and development tools
            gdb
            valgrind

            # Language server
            llvmPackages.clang-tools
          ];

          # Set up environment variables for OpenMP
          shellHook = ''
            # Export OpenMP include and library paths
            export CPATH="${llvmPackages.openmp.dev}/include:$CPATH"
            export LIBRARY_PATH="${llvmPackages.openmp}/lib:$LIBRARY_PATH"
            export LD_LIBRARY_PATH="${llvmPackages.openmp}/lib:$LD_LIBRARY_PATH"

            # OpenMP runtime settings
            export OMP_NUM_THREADS=4

            echo "OpenMP headers available at: ${llvmPackages.openmp.dev}/include"
            echo "OpenMP libraries available at: ${llvmPackages.openmp}/lib"
          '';
        };
      });
}
