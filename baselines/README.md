# Baseline implementation and provenance

The benchmark uses scalar forecasts at lead 24, not a 24-step output sequence.
All trainable adapters return one target-standardized number per example. See
`UPSTREAM_PROVENANCE.json` for pinned commits, upstream URLs, original file
hashes, benchmark changes, and public runtime hashes.

| Baseline | Pinned upstream | Public candidate handling |
|---|---|---|
| DLinear | Local scalar decomposition adapter (146 parameters) | Implemented in `phystok/models/adapters.py`; does not claim to redistribute the official DLinear repository |
| Informer | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020/tree/29f2a739226a509202a092b464163da81fa74960) | Apache-2.0 core and license included |
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | Apache-2.0 core and license included |
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124) | MIT core and copyright/license included |
| TimeXer | [thuml/TimeXer](https://github.com/thuml/TimeXer/tree/76011909357972bd55a27adba2e1be994d81b327) | Source not redistributed: license at the pinned commit is unconfirmed |
| XLinear | [Zaiwen/XLinear](https://github.com/Zaiwen/XLinear/tree/af500c5478ef2a69ad36d04ffcdfe04aae4483e2) | Source not redistributed: license at the pinned commit is unconfirmed |
| Persistence | Local past-only deterministic evaluation | No weights or training seeds |

The TimeXer and XLinear repositories were public when checked on 2026-11-08,
but no root license was found in their pinned trees. An absent `LICENSE` is not
an open-source grant. Obtain author permission or otherwise confirm your right
to use the source. Do not infer a license from another repository by the same
organization.

Once you have an authorized checkout at the exact commit, adapt the required
files locally:

```bash
python baselines/install_external.py --model timexer \
  --source-dir /path/to/authorized/TimeXer --confirm-source-use-rights
python baselines/install_external.py --model xlinear \
  --source-dir /path/to/authorized/XLinear --confirm-source-use-rights
```

These commands do not download code. They check each pinned source hash before
applying the benchmark's namespace and normalization changes, and check each
adapted hash afterward. The resulting package directories are local-only and
must remain ignored by Git. They are not part of the public release candidate.
The training adapter fails with an explanatory error if a required external
source is absent; it never substitutes a different architecture.

The active benchmark changes are documented in `docs/THIRD_PARTY_LICENSES.md`.
No official baseline task is claimed to be unchanged: 72-hour windows feed a
single city-level target at `t+24`. In particular, the Informer decoder has
36 observed timestamps followed by one zero-value query stamped at `t+24`.
