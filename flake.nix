{
  description = "C++ development environment with MPI and OpenMP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            gcc
            
            openmpi
            
            clang-tools
            gdb
          ];

          shellHook = ''
            export CC=gcc
            export CXX=g++
            
            # Ensure clangd can find all headers
            export CPATH="${pkgs.openmpi}/include:$CPATH"
            export CPLUS_INCLUDE_PATH="${pkgs.gcc.cc}/include/c++/${pkgs.gcc.cc.version}:${pkgs.gcc.cc}/include/c++/${pkgs.gcc.cc.version}/x86_64-unknown-linux-gnu:$CPLUS_INCLUDE_PATH"
            export LIBRARY_PATH="${pkgs.openmpi}/lib:$LIBRARY_PATH"
          '';
        };
      }
    );
}
