{
  description = "C++ development environment with MPI and OpenMP";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Use Clang-based stdenv (includes libc++ by default)
        stdenv = pkgs.clangStdenv;
      in
      {
        packages.default = stdenv.mkDerivation {
          pname = "cpp-env";
          version = "0.1";

          src = null;  # no source, we only want the environment/package

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            libcxx
            libgcc
            
            openmpi
            llvmPackages_latest.openmp
            
            vtk
            clang-tools
            yaml-cpp
            gdb
          ];

          shellHook = ''
            # Ensure we use clang from this environment
            export CC=${pkgs.clang}/bin/clang
            export CXX=${pkgs.clang}/bin/clang++

            # Help CMake find OpenMP
            export OpenMP_omp_LIBRARY="${pkgs.llvmPackages_latest.openmp}/lib/libomp.so"

            # Generate .clangd config for IDE support (clangd from clang-tools)
            cat > .clangd << EOF
CompileFlags:
  Add:
    - "-stdlib=libc++"
    - "-I${pkgs.llvmPackages_latest.libcxx.dev}/include/c++/v1"
  CompilationDatabase: build/
EOF

            echo "C++ dev environment ready (clangStdenv)"
            echo "CC=$CC  CXX=$CXX"
            echo ".clangd configuration written"
          '';
        };

        # Convenience: `nix develop` enters the same environment
        devShells.default = self.packages.${system}.default;
      }
    );
}
