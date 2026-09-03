# llb-rust: Puffer binding over the Rust kernel

Static PufferLib env that drives the play-llb-sim Rust kernel cdylib
instead of the managed GymHost/SHM path.

## Build

1. Build the cdylib: `cargo build --release -p play-llb-sim`
2. Stage PufferLib: `pufferlib-env/build-pufferlib.sh /path/to/pufferlib PROFILE`
   (copies ocean/llb-rust into the Puffer tree)
3. Build Puffer with the env: `cd /path/to/pufferlib && ./build.sh llb-rust --cpu`
   with `LLB_RUST_LIB=/abs/path/to/libplay_llb_sim.so` exported so the build
   and runtime agree on the cdylib location.

At runtime `LLB_RUST_LIB` must still name the cdylib (the binding dlopens it
lazily on first env creation).
