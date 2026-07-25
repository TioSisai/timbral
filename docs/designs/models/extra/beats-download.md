# BEATs Weight Acquisition (`scripts/extra/beats_dl.py`) Design

This document freezes the acquisition mechanism for the 15 official BEATs
checkpoints. See [`../encoders/beats.md`](../encoders/beats.md) for the
runtime component design, and [`beats-alignment.md`](beats-alignment.md) for
the alignment contract.

## Background and Decision

The official weights are published only via OneDrive public share links
(`1drv.ms` short links):

- The short links have migrated to SharePoint; the direct
  `api.onedrive.com` API returns 401, and the redirect target returns 403,
  so anonymous download over plain HTTP is not possible;
- An anonymous browser session can open the share page normally; while the
  page loads, OneDrive's `/items/` API response contains the file name, the
  exact byte count, and a temporary signed download URL
  (`@content.downloadUrl`), which can be downloaded via ordinary HTTP
  streaming.

The downloader is therefore implemented as an independent script,
`scripts/extra/beats_dl.py`, which uses Playwright Chromium anonymously to
resolve the signed URL and `requests` to perform the download. The
`src/timbral` runtime contains no download code at all:
`helpers.ensure_beats_checkpoint` only resolves paths and verifies SHA-256,
raising an error and printing this script's invocation command whenever a
file is missing.

## Script Responsibilities and Boundaries

- A single, standalone script: it does not import `timbral`, torch, or
  rootutils; only `requests` and the standard library are imported at the
  top level;
- `playwright.sync_api` is imported lazily inside the resolution function:
  environments without playwright (such as the audioal container) can
  safely import this module (for consistency testing), and a
  `ModuleNotFoundError` is naturally raised only when a download is
  actually attempted;
- It carries its own table of 15 pinned identities (see below), which
  mirrors the table in `helpers/beats.py`; consistency between the two is
  asserted by the default test suite;
- It performs no torch loading and no cfg validation — those are the
  responsibility of the runtime helpers.

## Pinned Identity Table

The table structure within the script (`entry -> BeatsDownloadTarget`):

| Field | Meaning |
|---|---|
| `share_url` | The OneDrive share link from the README table |
| `official_name` | The official file name returned by OneDrive; must match exactly after resolution |
| `size` | The exact byte count of the official file; must match exactly after resolution |
| `sha256` | The digest that must match once the download completes |

15 identities (`size`: 361499833 bytes for pretrained, 363145291 bytes for
fine-tuned):

| entry | `official_name` | SHA-256 |
|---|---|---|
| `beats_iter1` | `BEATs_iter1.pt` | `b5f4cc10bcbff63a437c695f33389e6411513b3f7d5cdae8fb62b5005f4a1fcd` |
| `fine_tuned_beats_iter1_cpt1` | `BEATs_iter1_finetuned_on_AS2M_cpt1.pt` | `e0e739e3670bfbb93c51adefb1d02981621397addc979d392aefd3dc53c22cab` |
| `fine_tuned_beats_iter1_cpt2` | `BEATs_iter1_finetuned_on_AS2M_cpt2.pt` | `2f3a7b65ab232c4f75570d4d17e21e5ebc34b3c40fe1a074f27d199e81354960` |
| `beats_iter2` | `BEATs_iter2.pt` | `81a23e00aa4878d7e8627ded87ea697fb347c8ceffed21223e0398ed0fa34ad8` |
| `fine_tuned_beats_iter2_cpt1` | `BEATs_iter2_finetuned_on_AS2M_cpt1.pt` | `3a120810c0f6dbfd50a7f48dc03ed077971a50cb2dbb7999695d5c700d03da45` |
| `fine_tuned_beats_iter2_cpt2` | `BEATs_iter2_finetuned_on_AS2M_cpt2.pt` | `08363b9b5eabeb47b0879c84145b27c603e7e50c116a633fa5b98ade119fc354` |
| `beats_iter3` | `BEATs_iter3.pt` | `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34` |
| `fine_tuned_beats_iter3_cpt1` | `BEATs_iter3_finetuned_on_AS2M_cpt1.pt` | `379369a41d0b3749f746cdcea8036de506cb3aedecce84de7db0a75fda2a4fe7` |
| `fine_tuned_beats_iter3_cpt2` | `BEATs_iter3_finetuned_on_AS2M_cpt2.pt` | `08374f1cbd49143900b351bc81cd307de386a11f8e609eb3862634e992068b55` |
| `beats_iter3_plus_as20k` | `BEATs_iter3_plus_AS20K.pt` | `8008b126bb5e8ab08912c60c58847ed676d32e64a5864c922356b7c2522fb2f8` |
| `fine_tuned_beats_iter3_plus_as20k_cpt1` | `BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt1.pt` | `2c366278dcf835e9bdefad4f7147b0edba4b940c59146fd05dc49a401fa82ff8` |
| `fine_tuned_beats_iter3_plus_as20k_cpt2` | `BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt2.pt` | `6d28b32bfa7bcaaf84ab834186581c2a360c6669e372e808d054cf0ef4d5c2d2` |
| `beats_iter3_plus_as2m` | `BEATs_iter3_plus_AS2M.pt` | `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34` |
| `fine_tuned_beats_iter3_plus_as2m_cpt1` | `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt` | `7f9362028ac6e5c049e8dc314d87e90e4f82a15a8e472deb56af55d7f9b34d6a` |
| `fine_tuned_beats_iter3_plus_as2m_cpt2` | `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | `e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c` |

The share links are pinned individually in the script (source: the last
three columns of the official README table):

| entry | `share_url` |
|---|---|
| `beats_iter1` | `https://1drv.ms/u/s!AqeByhGUtINrgcpmY7IHhgc9q0pT7Q?e=uQuisJ` |
| `fine_tuned_beats_iter1_cpt1` | `https://1drv.ms/u/s!AqeByhGUtINrgcpuRfRZmco2XulmFw?e=f2INHa` |
| `fine_tuned_beats_iter1_cpt2` | `https://1drv.ms/u/s!AqeByhGUtINrgcpyMlTmnRh0Wp_Qgg?e=sgzv8H` |
| `beats_iter2` | `https://1drv.ms/u/s!AqeByhGUtINrgcpwwEGgUyiI-jQyQw?e=1rP1RI` |
| `fine_tuned_beats_iter2_cpt1` | `https://1drv.ms/u/s!AqeByhGUtINrgcp4l547zKa7xPqy8w?e=rsLdPr` |
| `fine_tuned_beats_iter2_cpt2` | `https://1drv.ms/u/s!AqeByhGUtINrgcp5APbt_2bdIQvX0w?e=2cd2ry` |
| `beats_iter3` | `https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5` |
| `fine_tuned_beats_iter3_cpt1` | `https://1drv.ms/u/s!AqeByhGUtINrgcplb48ll1zIt82eWQ?e=XyxrX7` |
| `fine_tuned_beats_iter3_cpt2` | `https://1drv.ms/u/s!AqeByhGUtINrgcptb4S-CeJnlJGtZA?e=2FyDy3` |
| `beats_iter3_plus_as20k` | `https://1drv.ms/u/s!AqeByhGUtINrgcpvdNz8-aYim60CIg?e=53V8pg` |
| `fine_tuned_beats_iter3_plus_as20k_cpt1` | `https://1drv.ms/u/s!AqeByhGUtINrgcp2YHUCT1uZx2Kysw?e=nvu1Dw` |
| `fine_tuned_beats_iter3_plus_as20k_cpt2` | `https://1drv.ms/u/s!AqeByhGUtINrgcp092af0h7P3kXKFA?e=kUkPhN` |
| `beats_iter3_plus_as2m` | `https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf` |
| `fine_tuned_beats_iter3_plus_as2m_cpt1` | `https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC` |
| `fine_tuned_beats_iter3_plus_as2m_cpt2` | `https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea` |

## CLI

```bash
python scripts/extra/beats_dl.py \
    --dest /path/to/dir \
    [--entries beats_iter1 fine_tuned_beats_iter3_cpt2 ...] \
    [--workers 3]
```

- `--dest`: required, the download destination directory (created
  automatically);
- `--entries`: optional, `choices` are the 15 entry names, defaulting to
  all of them;
- `--workers`: number of parallel downloads, default 3.

Files are saved on disk as `<entry>.pt` (consistent with the helpers'
lookup convention), not under their official file names. If any file
fails, the errors are aggregated and the script exits with a non-zero code.

## Download Flow

The structure follows the reference implementation (anonymous Playwright
resolution + requests resumable download):

1. A single headless Chromium instance sequentially resolves the selected
   entries: it opens `share_url` and captures `name`, `size`, and
   `@content.downloadUrl` from the `/items/` response, retrying up to 3
   times;
2. The resolved result is strictly validated: `name == official_name` and
   `size == the pinned size`; any mismatch fails immediately (guarding
   against upstream content changes or link mismatches);
3. A `ThreadPoolExecutor` downloads in parallel into `<entry>.pt.part`:
   `Range`-based resumption, 8 MiB chunks, exponential-backoff retries (up
   to 5 times), and periodic progress output;
4. On completion, the byte count and SHA-256 are verified, after which the
   file is atomically `replace`d to `<entry>.pt`; if the byte count is
   insufficient, the `.part` file is kept for resumption; if the SHA-256
   does not match, verification info is printed, the `.part` file is
   deleted, and the download restarts from scratch immediately (counted
   toward the retry budget, failing once exhausted);
5. If the target file already exists: it is skipped if the SHA-256
   matches, and an error is raised without overwriting if it does not;
6. If the `.part` file is larger than expected: an error is raised without
   overwriting.

## Runtime Environment

- Requires `playwright` (including Chromium) and `requests`;
- The audioal container has no playwright: importing this script is safe,
  and a `ModuleNotFoundError` is naturally raised only when a download is
  actually attempted;
- The script does not depend on environment variables such as
  `HF_HUB_CACHE`; all directories must be passed explicitly.

## Test Requirements

`tests/scripts/test_beats_dl.py`, part of the default suite (run under
audioal), with no network access and no playwright triggered:

- Loads the script module by file path via `importlib`;
- Identity-table consistency: the script's entry set exactly matches
  `timbral.models.helpers.beats.BEATS_CHECKPOINTS`, with SHA-256 digests
  equal value-for-value; the correspondence between `official_name` and
  the fine-tuned flag is correct (a file name contains `finetuned` if and
  only if the helpers mark it as fine-tuned);
- The download function is empirically tested against a local in-thread
  HTTP server (with a self-implemented `Range` support): a fresh download,
  `.part`-based resumption, the 416/complete-file paths, rejection on
  byte-count mismatch, rejection when `.part` is oversized, automatic
  re-download after deleting `.part` on SHA-256 mismatch (failing once
  exhausted under persistent corruption, succeeding after recovery), and
  atomic renaming;
- An existing target file whose digest matches is skipped; a mismatched
  digest raises an error and does not overwrite.

The real OneDrive path is not written as a pytest test: during the
implementation phase, the script is run directly in an environment with
playwright, downloading at least one entry to `$TMPDIR` and verifying its
SHA-256; the result is recorded in the empirical-results section of
[`beats-alignment.md`](beats-alignment.md).

## Dependency Boundary

Running the script depends on `playwright` and `requests`; neither becomes
a `timbral` runtime dependency. The test side depends only on the standard
library and the existing test infrastructure.
