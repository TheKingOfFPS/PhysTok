# Third-party model code review

Checked on 2026-11-08 against the pinned upstream repositories used by the frozen
V2 benchmark. This report distinguishes upstream permission from ownership of
the authors' original PhysTok code; the repository-wide license remains an
author approval decision.

## Redistributed cores

- **Informer**: [Apache-2.0 at pinned commit](https://raw.githubusercontent.com/zhouhaoyi/Informer2020/29f2a739226a509202a092b464163da81fa74960/LICENSE).
  Six runtime Python files are included. Imports were namespaced in V2 and
  relocated to `phystok.models.vendor` for this candidate. Changed files carry
  a modification notice. The complete upstream license accompanies the code.
- **PatchTST**: [Apache-2.0 at pinned commit](https://raw.githubusercontent.com/yuqinie98/PatchTST/204c21efe0b39603ad6e2ca640ef5896646ab1a9/LICENSE).
  The three supervised backbone/layer files are included, with the complete
  upstream license and namespace-change notices.
- **iTransformer**: [MIT at pinned commit](https://raw.githubusercontent.com/thuml/iTransformer/c2426e68ca13f74aaec08045c5c724d8ad328124/LICENSE),
  copyright 2022 THUML @ Tsinghua University. Five runtime Python files are
  included with the original license. V2 changed input normalization from
  `x_enc /= stdev` to `x_enc = x_enc / stdev` to avoid an autograd version
  conflict, and guarded the unused optional Reformer import. These changes and
  the namespace relocation are recorded per file in the provenance manifest.

License copies reside both alongside the upstream runtime cores and in
`licenses/third_party/`. No upstream `NOTICE` file is part of the pinned file
inventory; the release owner should retain any additional notice supplied by
upstream or rights holders before publication. Original copyright/comments in
the source files are preserved. Code obtained through the official repositories
is attributed to its authors, not presented as PhysTok original work.

## External sources not redistributed

The pinned [TimeXer tree](https://github.com/thuml/TimeXer/tree/76011909357972bd55a27adba2e1be994d81b327)
and [XLinear tree](https://github.com/Zaiwen/XLinear/tree/af500c5478ef2a69ad36d04ffcdfe04aae4483e2)
do not show a root license file, and their pinned `/LICENSE` raw URLs returned
404 on this review. The permission to redistribute their source is therefore
**unconfirmed**. Complete source for these two cores is excluded from the
candidate. This is a concrete limitation for a full seven-model reproduction
until the researcher obtains the code and confirms use rights.

`baselines/install_external.py` provides the exact file-hash checks and local
adaptation method for an authorized checkout. TimeXer changes are namespace
imports, algebraically identical out-of-place normalization, and a guard for the
unused Reformer module; XLinear changes are algebraically identical
out-of-place normalization. The installer verifies adapted file hashes against
the actual frozen implementation. It does not grant permissions or download,
mirror, publish, or relicense upstream code.

## Local implementations and original assets

- The 146-parameter DLinear scalar adapter is local code expressing a moving
  average/decomposition and two linear maps. It is not a copy of the official
  DLinear training repository. The paper citation still credits DLinear.
- The Full PhysTok architecture snapshot is copied exactly, without replacing
  it by a later control. Its source ownership and the original scalar adapters
  require author/institution confirmation before choosing MIT or Apache-2.0.
- Delayed, Strict Late Fusion A, and Plain-State control modules preserve their
  frozen mathematics and initialization order; only package imports change.
- Model weights, source data, preprocessing output, prediction Parquet, server
  scripts and original operational logs are not distributed by this component.

The first-party repository license does not license third-party datasets or
expand the permissions of the upstream software. Full upstream file URLs,
commits, hashes and per-file changes are in `baselines/UPSTREAM_PROVENANCE.json`.
