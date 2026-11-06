# Anna Android ARM64 Port — Stage 1 (NEON + Dotprod)

## Scope
- Preserve Anna's original Rust engine and network format.
- Add AArch64 NEON kernels for the legacy 16-bit NNUE and v3 NNUE kernel sets.
- Add an AArch64 Dotprod implementation for v3's unsigned-u8 by signed-i8 dense L1 dot products.
- Keep the existing x86-64 AVX2 path unchanged and retain scalar reference kernels.
- Add Android NDK API 24 workflow outputs for ARM64 NEON baseline and ARM64 Dotprod.

## SIMD design
- Legacy NNUE: NEON wrapping i16 accumulator updates and widened i32 SCReLU output accumulation.
- v3 NNUE: NEON accumulator row updates, pairwise activation packing, dense-layer FP32 kernels and baseline widened u8-by-i8 dot products.
- Optional policy network: a NEON dot kernel is provided for its full 0..256 activation range; it remains scalar on other non-AVX2 architectures.
- Dotprod v3 L1: inline-assembly SDOT handles the low seven bits of u8 inputs. A masked signed-weight correction adds 128 times each weight where the input high bit is set. This preserves the full u8 API; the production pairwise activation currently emits 0..127. Inline assembly avoids depending on a recent Rust std::arch intrinsic stabilization.
- Scalar implementations remain the reference. Unit tests compare active SIMD kernels against scalar, including full-range u8 dot-product inputs.

## Android integration
- Cross-build target: `aarch64-linux-android` (Android API 24 linker/compiler). `engine/build.rs` now accepts multiple `FATHOM_MARCH` options as separate compiler arguments so the C tablebase code gets the intended `-march` and `-mtune` flags.
- Baseline build explicitly disables Dotprod; Dotprod build explicitly enables it.
- Rust targets use the generic AArch64 CPU plus only the required SIMD feature flags, avoiding accidental Cortex-A76-only instructions on a big.LITTLE Android phone. Vendored Fathom is tuned for Cortex-A76 while its `-march` matches the selected baseline/Dotprod ISA.
- x86-only BMI2/AVX2 routines remain compile-time gated; the AArch64 path uses the portable magic-bitboard fallback. TT, NNUE-row and correction-history prefetch paths use AArch64 `PRFM`.
- The Android workflow compiles the target-specific Rust library test harness for both variants and publishes it beside each engine binary, so the user can execute the NEON/Dotprod scalar-differential tests directly on the phone in Termux.

## Verification status
This source-stage port has not yet been compiled with Rust or the Android NDK in this environment. GitHub Actions must pass before claiming build success. The workflow fetches the upstream default network once, runs `netcheck --strict`, UCI, bench and perft checks on a host reference build, and passes the exact validated network bytes plus checksum to both Android variants. It cross-compiles both Android executables and their Rust library test harnesses, verifies the target ELF and NEON/SDOT instructions, and uploads the binaries plus diagnostic outputs. It does not try to execute Android/Bionic binaries on the x86-64 hosted runner. Run the matching `*-simd-tests` artifact on the phone in Termux to execute the scalar-differential tests on actual ARM64 hardware, then run the engine's UCI/bench/perft commands using the included phone instructions. The source snapshot did not contain `nets/default.bin`; network acquisition is therefore an explicit CI step and a failed/mismatched model check stops the build. Device runtime and playing-strength tests remain pending.
