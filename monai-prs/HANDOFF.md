# MONAI Contribution Handoff — Fresh Issues (#5939, #5461)

> **How to use this file:** Paste this entire document as the task brief into a **new
> Claude Code session that is scoped to your fork `vishnukannaujia/MONAI`** (see
> "Session requirements" below). It contains everything needed to implement two MONAI
> contributions and open the PRs — no other context required.

---

## Why this handoff exists

The previous session was scoped only to `vishnukannaujia/claude-code`, so it could not
fork, push to, or open PRs against `Project-MONAI/MONAI`. It also discovered that the two
originally-planned issues were **already covered by competing PRs** (see "Do NOT work on"
below), so we pivoted to two fresh, uncontested issues.

## Session requirements (do this before feeding the brief)

1. Fork `Project-MONAI/MONAI` on GitHub → creates `vishnukannaujia/MONAI`.
2. Give the Claude GitHub App access to `vishnukannaujia/MONAI`
   (GitHub → Settings → Applications → Claude → Configure → add the fork).
3. Start a new session at claude.ai/code with **`vishnukannaujia/MONAI`** as the repo.

## Identity / commit rules (STRICT)

- Author all commits as: `Vishnu Kannaujia <vishnu.kannaujia@gmail.com>`.
- DCO sign-off is mandatory: `git commit -s` (the DCO bot fails the PR otherwise, and the
  email must match the GitHub account).
- **No Claude/AI attribution anywhere** — not in commit messages, PR titles/bodies, code
  comments, or test files.
- Target branch for all PRs is **`dev`** (NOT `main`).

## Setup

```bash
git clone https://github.com/vishnukannaujia/MONAI.git && cd MONAI
git remote add upstream https://github.com/Project-MONAI/MONAI.git
git fetch upstream
git config user.name "Vishnu Kannaujia"
git config user.email "vishnu.kannaujia@gmail.com"
# Base every feature branch off the latest upstream dev:
git checkout -b <feature-branch> upstream/dev
```
Reference point: upstream/dev tip at handoff time was
`eccefc57550b111ed781d82249dfe77872a0e918` ("Release 1.6 Doc Updates (#8882)").

## Before opening each PR
Comment on the issue ("I'd like to take this — PR incoming") so a maintainer can assign it.

## Local checks (must pass before push)
```bash
python -m pip install -U -r requirements-dev.txt
./runtests.sh --autofix          # black, isort, ruff
./runtests.sh --quick --unittests
```
If only formatting fails in CI, comment `/black` on the PR to auto-fix remotely.

---

## Do NOT work on (already have competing open PRs — verified 2026-06-15)

- **#7980** (writer install hint) → open PR **#8795** already in review (+ #8761). HELD.
- **#7437** (LoadImage reader raise) → **six** open PRs: #8823, #8796, #8771, #8768, #8522.
  HELD.
- **#6119** (Windows BUILD_MONAI docs) → open PR #8906.
- **#5537** (CLI checkpoint inspector) → open PR #8830.

> The fully-implemented + tested work for #7980 and #7437 is preserved as git patches in
> `vishnukannaujia/claude-code` on branch `claude/monai-2-prs-handoff-046h0d` under
> `monai-prs/patches/pr1/` and `monai-prs/patches/pr2/`. Revive only if the competing PRs
> stall and a maintainer signals our angle is wanted.

---

# PR A — Issue #5939: BendingEnergyLoss numerical stability

**Issue:** "BendingEnergyLoss" — open, unassigned, labels: Contribution wanted, Feature
request. **No competing PRs** (verified).

Suggested branch: `5939-bending-energy-second-order-kernel`

### Problem
`monai/losses/deform.py` computes the bending energy's **second-order** derivatives by
applying the first-order central-difference helper `spatial_gradient()` **twice**.

`spatial_gradient(x, dim)` returns `(x[i+1] - x[i-1]) / 2`. Applying it twice yields:
`(x[i+2] - 2·x[i] + x[i-2]) / 4` — i.e. a wide `[1, 0, -2, 0, 1]/4` stencil spanning 4
grid points. Consequences:
- Less accurate / more prone to artifacts than the standard compact second-difference.
- Forces the constraint "all spatial dims > 4" (see `forward()` validation).

The issue requests:
- For pure second derivatives ∂²/∂xᵢ²: use the compact `[1, -2, 1]` kernel
  (`x[i+1] - 2·x[i] + x[i-1]`), spanning 3 points.
- For mixed partials ∂²/∂xᵢ∂xⱼ: use a compact finite-difference scheme (e.g. forward
  difference of forward difference) rather than central-of-central. Reference:
  Pavel Holoborodko's finite-difference notes (cited in the issue).

### Files
- Source: `monai/losses/deform.py` — class `BendingEnergyLoss.forward()` (≈ lines 74-125),
  plus the shared helper `spatial_gradient()` (≈ lines 20-44).
- Test: `tests/losses/deform/test_bending_energy.py`.

### Implementation guidance
1. Add a `second_order_gradient`/`spatial_gradient(..., order=2)` style helper (or inline
   it) implementing `[1, -2, 1]` for the pure second derivative along one axis.
2. For mixed partials, compose first-order differences in a compact way (forward-then-
   forward, or central with a 3-point footprint), not central-of-central.
3. Preserve the existing public API: `__init__(normalize, reduction)` and `forward(pred)`,
   and keep the `normalize` behavior (scaling by spatial sizes — see current lines 100-116).
4. **Relax the validation** "all spatial dimensions must be > 4" to the new minimum the
   compact kernel needs (likely **> 2**), and update the matching `test_ill_shape` cases.
   The `DiffusionLoss` in the same file already uses "> 2" as a precedent.

### Expected test impact (IMPORTANT)
Changing the stencil **changes the numeric loss values**, so the hard-coded expectations in
`TEST_CASES` (e.g. `4.0`, `100.0`) WILL change. After implementing:
- Recompute the expected values analytically for the `x = arange(0,5)**2` style inputs.
  For `f(x)=x²`, the exact second derivative is constant `2`; the `[1,-2,1]` kernel returns
  exactly `2` everywhere (vs the old wide stencil's value), so derive each case's expected
  energy by hand and update `TEST_CASES`.
- Keep the zero-energy cases (constant field, linear field) at `0.0` — `[1,-2,1]` gives 0
  for constant and linear inputs, so those should still hold.
- Update `test_ill_shape` to reflect the new minimum spatial size.
- In the PR body, **show the math** justifying the new expected values (reviewers will
  scrutinize a change to established loss numerics).

### Verify
```bash
python -m pytest tests/losses/deform/test_bending_energy.py -v
# Also sanity-check nothing else imports/relies on the >4 constraint:
python -m pytest tests/losses -k "deform or bending or diffusion" -v
```

### PR
- Title: `Use compact [1,-2,1] kernel for BendingEnergyLoss second derivatives`
- Body: "Fixes #5939." + 3-4 line summary: replace central-of-central second differences
  with the compact `[1,-2,1]` (and a compact mixed-partial scheme), improving numerical
  stability and relaxing the spatial-size constraint from >4 to >2; updated unit-test
  expectations with derivation. Target `dev`.

---

# PR B — Issue #5461: WSIReader / test_wsireader unclosed file handles

**Issue:** "tests.test_wsireader unclosed file" — open, unassigned, labels: bug,
Contribution wanted. **No competing PRs** (verified).

Suggested branch: `5461-wsireader-close-file-handles`

### Problem
Running the WSI reader tests emits `ResourceWarning: unclosed file <_io.FileIO ...>` /
`BufferedReader` for a temp TIFF (e.g. `temp_CMU-1.tiff.tiff`). Root cause: the TiffFile
backend opens file handles that are never closed.

In `monai/data/wsi_reader.py`, `TiffFileWSIReader.read()` (≈ line 1491-1500) does:
```python
for filename in filenames:
    wsi = TiffFile(filename, **kwargs_)   # line ~1497 — opens a file handle
    wsi_list.append(wsi)
return wsi_list if len(filenames) > 1 else wsi_list[0]
```
The returned `TiffFile` objects hold open OS file handles and nothing closes them. The
CuCIM/OpenSlide backends may exhibit a similar pattern — check `CuCIMWSIReader` (class at
≈ line 741) and `OpenSlideWSIReader` (≈ line 1024).

### Files
- Source: `monai/data/wsi_reader.py` (TiffFile backend, and possibly the base reader).
- Test: `tests/utils/enums/test_wsireader.py` (note: file moved from the old
  `tests/test_wsireader.py` path referenced in the issue).

### Implementation guidance (decide after reproducing)
First **reproduce** with warnings enabled:
```bash
python -m pytest tests/utils/enums/test_wsireader.py -W error::ResourceWarning -v
# (requires the TiffFile backend installed: pip install tifffile imagecodecs)
```
Then choose the cleanest fix that satisfies reviewers:
- **Test-hygiene fix (smallest):** ensure each opened WSI object is closed in the tests
  (e.g. call `wsi.close()` in a `tearDown`/`finally`, or use it as a context manager). The
  issue title frames this as a test problem, so this may be the expected scope.
- **Reader-level fix (more robust):** give the reader a way to release handles — e.g. a
  `close()` method on the WSI object usage path, or close the `TiffFile` once the data has
  been materialized where the API allows. Be careful: `read()` is documented to return an
  open WSI object that later `get_data`/patch calls use, so don't close prematurely. If you
  add reader-level cleanup, cover it with a focused test.

Keep the change minimal and well-scoped; confirm the warning is gone and the full
`test_wsireader.py` still passes.

### Verify
```bash
python -m pytest tests/utils/enums/test_wsireader.py -W error::ResourceWarning -v
```

### PR
- Title: `Close WSIReader file handles to fix unclosed-file warnings in tests`
- Body: "Fixes #5461." + 3 line summary of where handles leaked and how they're now
  closed. Target `dev`.

---

## Final checklist for the new session
- [ ] Both branches based on `upstream/dev`, independent of each other.
- [ ] Commits authored as Vishnu Kannaujia, signed off (`git commit -s`), no AI attribution.
- [ ] `./runtests.sh --autofix` and `--quick --unittests` clean.
- [ ] Each issue commented on before opening its PR.
- [ ] Two PRs opened against `Project-MONAI/MONAI:dev`, each with "Fixes #<n>".
- [ ] For #5939: PR body shows the math justifying updated test expectations.
