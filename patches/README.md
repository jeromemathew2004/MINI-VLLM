# Patches to `mini-flash-attention`

`requirements.txt` installs the attention backend from
`w4096/mini-flash-attention`. Upstream has a correctness bug that mini-vllm hits
in normal operation; until it is fixed there, a local build with the patch here
is required.

## `mini-flash-attention-decode-race.patch` — required

**Without it, decode is nondeterministic and wrong for batch widths >= 6.**

`flash_attention_fwd_split_kv_kernel` aliases two different things over the same
`extern __shared__` region. `warp_max_val` / `warp_expsum_val` (the cross-warp
softmax reduction) sit at `smem_data` offset 0, and so does `warp_output`.
Every warp reads the reduction values, then each warp writes 128 floats of
output starting at offset 0 — clobbering them. There was no barrier between the
reads and the write, so warp 0's output write races the other warps' reads.

The damage lands on the softmax normalisation, so it is a rescaled output rather
than a small perturbation: logits move by 10-30, and the emitted token changes.
It only manifests once occupancy lets warps drift out of lockstep, which is why
it presents as batch-size-dependent nondeterminism rather than a constant wrong
answer.

The fix is one `__syncthreads()` between the reduction and the output write.

Confirmed with `compute-sanitizer --tool racecheck`, which reported ~4,500
hazards per launch between `decode.cuh:512` (the write) and `:604` / `:613` (the
reads), and reports 0 after the patch.

Regression check: `python experiments/decode_determinism_check.py`.

## Applying it

The source tree used to build this host's backend is at `C:\Users\jerry\mfa-build`
(clone it somewhere with a short path — CUTLASS filenames overrun Windows
`MAX_PATH` from a deep directory).

```sh
cd /path/to/mini-flash-attention
git apply /path/to/mini-vllm/patches/mini-flash-attention-decode-race.patch
```

Then rebuild. On this host that is `C:\Users\jerry\mfa-build\build_win.bat`,
which pins CUDA 12.6 and MSVC toolset 14.44 and installs into the repo venv.
That script also depends on three Windows fixes to upstream's `setup.py`
(GCC-spelled flags, `lib64` vs `lib/x64`); those are described in `PROGRESS.md`
under Environment State and are not duplicated here, since they are
environment-specific while the patch above is a correctness fix for every
platform.

Verify after rebuilding:

```sh
python experiments/decode_determinism_check.py   # deterministic at every width
python experiments/verify_pass_gate.py           # GATE: PASS
```
