# FeynmanReducer

FeynmanReducer reduces Feynman integrals from a YAML topology definition. It
finds a master basis, constructs finite-field IBP reductions, and reconstructs
the resulting coefficients with FireFly.

## Requirements

- CMake 3.20 or newer
- GCC, Clang, or Apple Clang targeting a 64-bit platform with GNU C++20 and
  `unsigned __int128` support
- A POSIX/Unix-like system with pthreads
- yaml-cpp
- zlib (gzip reference inputs for `reduction_validate`)
- FLINT
- FireFly 2.0.3
- Singular 4.x with `primdec.lib`
- At least one symmetry backend: nauty with Traces, or bliss

The main library dependencies must be visible to `pkg-config`:

```bash
pkg-config --exists flint firefly
```

## Build

Configure and build the Release preset from the `Program/` directory (the release-source root):

```bash
cmake --preset default
cmake --build --preset default -j8
```

The executables are written to `build/FeynmanReducer` and `build/reduction_validate`.

Release builds are optimized for the build machine by default. To build a
portable binary instead, configure with native CPU optimization disabled:

```bash
cmake --preset default -DFR_NATIVE_OPTIMIZATION=OFF
cmake --build --preset default -j8
```

CMake detects nauty and bliss automatically and compiles the backends it
finds. Configuration fails if neither backend is available.

## Run

Run the bundled outer-massive double-box example with:

```bash
cd examples/double_box_outer_massive
../../build/FeynmanReducer
```

When `-i` is omitted, FeynmanReducer reads `config.yaml` from the current
directory. The general command-line interface is:

```text
FeynmanReducer [-i config.yaml] [-j N]
```

- `-i, --input PATH` selects the YAML input file.
- `-j, --threads N` overrides the thread count in the YAML file.
- `-h, --help` prints the command-line summary.

Relative paths in a configuration are resolved against the directory
containing that YAML file. Use the files under `examples/` as configuration
templates.

The bundled examples select the nauty backend. With a bliss-only build, change
`symmetry_backend` in the selected YAML file to `bliss`.

Results are written under `outputs/` next to the input YAML file:

```text
outputs/results.m
outputs/final_basis.txt
outputs/reduction_YYMMDDHHMMSS.log
outputs/firefly.log                 # only when FireFly emits a raw log
```

Coefficients retain polynomial factors and powers. After reconstruction, mixed
D/kinematic denominator factors produce a warning with default basis selection
and an error with explicit `d-separating`; the check is skipped when D is fixed.

`build/reduction_validate -i examples/double_box_outer_massive/config.yaml`
compares the output with static Kira references. The default reference is
`validation/kira_integrals.m`, falling back to `.m.gz` when plain text is absent.
Gzip input is decompressed in memory by zlib.
