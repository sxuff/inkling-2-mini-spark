# Clean-clone check

These files come from a run of the published scripts on the test GX10, from a fresh copy of this repository:

1. `scripts/download.py` checked a hard-linked copy of the checkpoint against the pinned revision: 13 files, sizes and LFS SHA-256.
2. `scripts/serve.sh q8` started with an empty cache. It was ready after 230 s, with the image already pulled.
3. `scripts/bench.py --compare results/ab/stock.json` gave 66.62 tok/s median, 9/9 output hashes equal to stock vLLM, canary 408, and the 2- and 8-image vision checks passed (`serve-q8-clean-clone.json`).

Step 3 of `log.txt`, and the bench after it, are specific to the test machine. Its long-running service had been promoted with the same kernels before notices were added to the file headers. The service was switched to the byte-identical repository files, then benched again at 66.38 tok/s with 9/9 hashes and vision passing (`live-after-kernel-sync.json`).
