# Dense HCF diagnosis — 2026-09-12

## Root cause

The supplied September 12 trace identifies an **SDK 2.10 local-simulator ingress
ring-buffer overflow**. Queued H2D uploads overwrite unread stream data after
100 MiB. The application launchers now work around this by bounding each upload
and waiting for device readback before submitting the next batch.

This is not an arithmetic or reduction-routing fault. No CSL kernel changes
were made as part of this fix.

## Evidence

The server halts at physical P2.1, cycle 265659, in the SDK demultiplexer.
`demux.csl:478` loads `num_receivers = 57135`; line 482 loads
`memcpy_optype = 13681`. Neither is a valid header for this upload. Line 376
passes `57134` to `filters/wse3.csl:21`, whose limit assertion halts execution.

The P1.1 adaptor's 36 incoming header words match the seeded weight matrix
**bit for bit**. They are data from PE row 49, starting at word 522488 of that
row's wire-ordered weight payload. Each W row has 524288 payload words and a
36-word header. Relative to the first W header, the displacement is:

```
49 * 524288 + 522488 + 50 * 36 = 26214400 words
26214400 * 4 = 104857600 bytes = 100 MiB
```

The adaptor interprets 32 FP16 weight bit patterns interleaved with zero upper
halves as receiver flags. Their prefix sum wraps to 57135, exactly the bad
receiver count in the demultiplexer trace. Run
`python3 MeshGEMVPatched/debug/evidence/verify_trace_wrap.py` to verify the match
against the supplied ZIP.

The matching SDK's `/cbcore/lib/libstreamer.so` allocates `0x6400000` bytes for
this ring. Its local send path accounts for the distance to the physical end of
the ring, but does not limit writes to unread capacity. A direct test of the
**unmodified native writer**, using isolated synthetic ring state, filled the
100 MiB buffer without consuming anything, then wrote one additional word.
The write returned successfully and replaced the first unread word.
`evidence/check_sdk_ring.py` and `evidence/native_ring.txt` preserve this
SDK-build-specific reproduction. It does not modify the installed SDK.

A separate 262144-element H2D mutation test read back the original values after
the NumPy source was changed immediately after blocking submission. Retaining
Python arrays is not the fix for the diagnosed native ingress overflow.

## Why matrix size and row chunking matter

FP16 H2D uses one uint32 stream word per value. Thus 4096x4096 W occupies
64 MiB of ingress payload; 4096x8192 W occupies 128 MiB. The latter exceeds the
100 MiB ring even before X and protocol headers. P=128 does not reduce the
matrix's total stream size.

`nonblock=False` H2D can return after the host has submitted data to the local
streamer, before the simulated device has consumed it. Sending all row chunks
without readbacks still queues the same total payload and can overwrite the
ring. This explains why the earlier row-chunking probe still failed.

## Implemented workaround

`MeshGEMVPatched/sim_transfers.py` sends at most 16 MiB of payload per batch, splitting
PE rows and, when needed, columns. After each batch it reads one uploaded value
from **every affected PE** with blocking D2H, and checks the returned bits. The
readback establishes device completion before another upload is submitted.
X uploads use the same path. The separate patched simulator launchers and the chunked probe
use this helper. Original MeshGEMV uploads are restored; select `--hcf-fix`
in Llama2Projections to activate the workaround. All checkpoints precede the timed kernel region.

The probe and supervisor default to fenced uploads. `--no-fence-h2d` retains the
old schedule only for reproducing the SDK bug. `--suppress-simfab-trace` avoids
huge trace archives during successful validation runs.

## Validation

- Exact 36-word server-trace match and 100 MiB displacement: passed.
- Native SDK ring-overwrite reproduction: passed (confirmed the bug).
- Three upload regression tests: passed. Cover row and column splits, partial
  rectangles, raw FP16 bit preservation, completion for all destinations,
  payload bounds, and stopping on a bad checkpoint.
- Fixed diagnostic: P=4, 256x512, local tile 64x128 (same local tile as the
  failing P=64 case), one warmup and one measured repeat: passed. All 512
  output values on one replicated PE row were checked; max relative error
  0.0019, mean 2498 cycles, max 2504 cycles.

- Updated production `launch_sim_llama.py` on that isolated artifact: passed
  numerical validation of every replicated PE row, max relative error 0.0019.

The complete P=64, 4096x8192 fixed run still needs server validation. The native
bug is reproduced locally; a successful full-size result is not yet claimed.


## Layout update — September 13

The patched dense implementation, upload regression tests, and debug tools now
live under `MeshGEMVPatched/`. `Llama2Projections/run.sh` activates the patch with
`--hcf-fix` and persists that selection in the run configuration. Dense and fused
have separate patched entry points; fused uses the original low-rank CSL kernel.
The original dense launcher uses unfenced uploads again. All copied dense CSL
files were checked byte-for-byte against the original folder.
