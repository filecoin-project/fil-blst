Rust binding for fil-blst (Filecoin blast)
==========================================

Rust binding for a library to accelerate SNARK verification for the Filecoin network using the blst BLS12-381 performance library.

Building
--------

    cargo build


Publishing the crate
--------------------

You need to set the `FIL_BLST_SRC_DIR` to the root (absolute path) of the `fil-blst` checkout (not to the `rust` subdirectory).

    FIL_BLST_SRC_DIR=${PWD}/.. cargo publish
