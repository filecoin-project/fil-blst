# zkblst (Zero knowledge blast)
A library for zero knowledge related computations using the blst BLS12-381 performance library

# Building

git submodule update --init --recursive

cd blst
git checkout d31b0b5
./build.sh
cd ..

cd cpp
g++ -I../blst/bindings zkblst.cpp main.cpp ../blst/libblst.a
./a.out

To run the rust test you need to first edit build.rs to include libblst.a.
sudo apt install clang
cd rust/
cargo test
