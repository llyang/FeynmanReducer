# FeynmanReducer

FeynmanReducer reduces Feynman integrals from a YAML topology definition. It finds a master basis, constructs finite-field IBP reductions, and reconstructs the resulting coefficients with FireFly.

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

Release builds are optimized for the build machine by default. To build a portable binary instead, configure with native CPU optimization disabled:

```bash
cmake --preset default -DFR_NATIVE_OPTIMIZATION=OFF
cmake --build --preset default -j8
```

CMake detects nauty and bliss automatically and compiles the backends it finds. Configuration fails if neither backend is available.

## Run

Run the bundled outer-massive double-box example with:

```bash
cd examples/double_box_outer_massive
../../build/FeynmanReducer
```

When `-i` is omitted, FeynmanReducer reads `config.yaml` from the current directory. The general command-line interface is:

```text
FeynmanReducer [-i config.yaml] [-j N]
```

- `-i, --input PATH` selects the YAML input file.
- `-j, --threads N` overrides the thread count in the YAML file.
- `-h, --help` prints the command-line summary.

Relative paths in a configuration are resolved against the directory containing that YAML file. Use the files under `examples/` as configuration templates.

`targets_file` accepts either a single file name or a list of file names. Each file can be a plain-text file (`.txt`) or a YAML file (`.yaml` or `.yml`). The plain-text files contain one headed integral per entry, such as `F[2,1]`. Every integral head, including a component in a combination YAML file, must equal the configured `integral_header`; a mismatched head is rejected instead of being silently rewritten. The YAML files are designed to represent targets that are linear combinations of integrals:

```yaml
combinations:
  - name: amp[1]
    terms:
      - {coefficient: "s", integral: "F[2,1]"}
      - {coefficient: "1", integral: "F[1,1]"}
```

Each `integral` must be a scalar headed expression in the same form as a text target. Component integrals share one reduction kernel, while FireFly reconstructs the contracted named outputs. Symbolic combination terms must have a common scale degree. With `d-separating` selection, every component integral is validated; the contracted named coefficients are not themselves required to remain D-separating. `reduction_validate` currently supports integral-only target files and rejects configurations containing combination YAML explicitly.

Targets may also carry an even additive dimension shift:

```text
F[2,{1,1}]
F[-4,{2,1}]
```

`F[k,{a1,a2,...}]` denotes the integral in dimension `d+k`. Positive, zero, and negative even `k` are accepted; odd shifts are rejected. The legacy `F[a1,a2,...]` syntax is unchanged, and `k=0` is written back in that legacy form. In combination YAML, shifted integrals use the same syntax as a quoted scalar, for example `integral: "F[2,{1,1}]"`. Master integrals remain in dimension `d`. Both default and `d-separating` reduction support shifted targets, including targets with numerator/ISP indices.

Master indices must currently be nonnegative. Negative-index targets are supported through projected top-LP reduction, but choosing a negative-index master will require future extended-LP basis coordinates, replay/reconstruction support, and an extended differential-equation path. Dimension-shifted masters are also unsupported.

`reduction_validate` does not yet have a cross-dimension Kira comparison model and explicitly rejects configurations containing dimension-shifted targets.

Master-integral differential equations can be generated directly from the
Lee--Pomeransky polynomial by adding:

```yaml
differential_equations: true
```

The program differentiates with respect to every free kinematic parameter
(but not `d`), converts each derivative to a linear combination of `d+2`
integrals, and reduces those combinations to the final `d`-dimensional master
basis. Parameters fixed by `numerics` are not differentiated. If all
kinematic parameters are fixed, the configuration is rejected.

`targets_file` remains optional for this mode. When it is present, ordinary
reductions and differential equations share one kernel. When it is absent,
the run is DE-only and does not implicitly read `targets.txt`. Both `default`
and `d-separating` basis selection are supported. D-separating discovery always
uses all symmetry-inequivalent top-sector integrals with both dots on one
propagator, i.e. exactly one index equal to three; user targets do not participate
in the search. After selection, ordinary and shifted
targets, every nonzero combination component, and derivative source integrals for
the selected basis are checked against that fixed basis. A failure terminates the
run without another search. D-separating constrains those source integrals, not the
contracted entries of the resulting connection matrices.

Three focused examples under `examples/` exercise these interfaces directly:

- `linear_combination/` reads the 22 original targets from `targets.txt`, then two mixed-mass-dimension named combinations from `targets.yaml`; this also demonstrates mixed text/YAML target files and stable file-order output.
- `dimension_shift/` combines the 22 original targets and eight representative dotted/ISP targets in dimensions `d+2` and `d-2` in one `targets.txt`.
- `differential_equations/` uses the outer-massive double box to reduce the 22
  original targets and generate the `msq`, `s`, and `t` connection matrices in
  the same run.

The bundled examples select the nauty backend. With a bliss-only build, change `symmetry_backend` in the selected YAML file to `bliss`.

Results are written under `outputs/` next to the input YAML file:

```text
outputs/results.m
outputs/final_basis.txt
outputs/differential_equations.m   # when differential_equations is enabled
outputs/reduction_YYMMDDHHMMSS.log
outputs/firefly.log                 # only when FireFly emits a raw log
```

Coefficients retain polynomial factors and powers. After reconstruction, standalone integral outputs with mixed D/kinematic denominator factors produce a warning with default basis selection and an error with explicit `d-separating`; named combinations and differential-equation entries are excluded because their source integrals are the validation unit. With explicit `d-separating`, source-only runs are covered by the basis-selection certificate. With default selection and no standalone integral outputs, this diagnostic is skipped. The check is also skipped when D is fixed.

`differential_equations.m` is a Mathematica replacement list from each free
kinematic parameter to its connection matrix, for example
`{s -> {{...}}, t -> {{...}}}`. Matrix rows and columns both follow
`final_basis.txt`. In a DE-only run `results.m` is the empty replacement list.
`reduction_validate` does not yet validate differential matrices and rejects
such configurations explicitly.

`build/reduction_validate -i examples/double_box_outer_massive/config.yaml` compares the output with static Kira references. The default reference is `validation/kira_integrals.m`, falling back to `.m.gz` when plain text is absent. Gzip input is decompressed in memory by zlib.
