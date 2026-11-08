# Ownership and license decision

## Current decision

No license for project-authored code has been granted in this local candidate. `LICENSE` records that status; `licenses/candidates/` contains unmodified proposal texts. Publication requires the authors' approval of copyright ownership, third-party compliance and the selected license. A manuscript byline does not, by itself, establish ownership of every source file or authorize institutional IP release.

## Rights inventory

| Material | Evidence and treatment | Decision needed |
|---|---|---|
| PhysTok, scalar adapters, preprocessing, orchestration, statistics and documentation | Extracted from the authors' research implementation or written for this candidate; source provenance is retained. No employment, collaboration or institutional assignment agreement was supplied. | Confirm the appropriate copyright holder(s), institutional authority and permission to publish. |
| Informer, PatchTST, iTransformer components | Upstream commits, source hashes and license notices are recorded in `docs/THIRD_PARTY_LICENSES.md` and `baselines/`. | Keep the corresponding notices and modification records when distributing. |
| TimeXer and XLinear cores | No explicit redistribution license was established at the pinned source versions. Full sources are excluded, even though original research used them. | Obtain applicable permission or an explicit upstream license before redistributing them. The local installer does not grant rights. |
| Aggregate error tables and sufficient statistics | Aggregate research outputs; no observations, individual predictions or timestamped residuals are included. | Authors must approve public release of the selected aggregates and their reuse terms. This does not confer rights to the underlying data. |
| City/fold metadata | Selected city identities, locations and assignments; no concentration or weather measurements. | Confirm this metadata inventory is approved for publication. Third-party station registry files are excluded. |
| Paper, figures, datasets and weights | The full manuscript, figures, original/processed data, predictions and weights are not included. | No release grant is inferred from their availability in the private project. |

## Candidate comparison

**Apache-2.0 is the recommended candidate for author review.** It provides an express patent license and patent-litigation termination terms, and its attribution/modification requirements are a natural fit for this research code with existing Apache-licensed components. It does not settle ownership, grant trademark rights, or authorize third-party datasets. See the [official Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0).

**MIT is the shorter alternative.** It permits broad reuse subject to retaining the copyright and permission notice and includes a warranty disclaimer. Its text does not contain Apache-2.0's express patent-grant provisions. Choosing MIT for original work would not remove the Apache obligations of bundled third-party files. See the [OSI MIT text](https://opensource.org/license/mit).

Neither candidate requires an invented copyright holder. Once approved, replace the pending `LICENSE` notice with the chosen final text, insert the confirmed holder(s) where appropriate, retain upstream notices and add a clear results/metadata reuse notice. The candidate texts themselves are not grants for the rest of the repository.

## Before publication

The authors should confirm code and aggregate ownership, institutional/collaboration approval, the manuscript's allowed public status, the selected code/results terms, and whether the two external source cores may be redistributed. Those approvals have not been inferred from this code-preparation task. No GitHub remote, push, public Release, or data mirror is created by the candidate workflow.
