# Private repository staging — November 9, 2026

The author explicitly authorized creation and upload of a **private** GitHub repository on November 9, 2026:

https://github.com/TheKingOfFPS/PhysTok

This authorization covers private storage and collaboration. Public release, the final first-party license and aggregate-data reuse terms remain pending author approval. The pending `LICENSE` notice is retained. No public Release or dataset upload is part of this operation.

The scientific sources and results are unchanged from the audited local candidate, initial commit `f660194a9e40ef3d8c2c875c6600043e52bd00a0`. This update adds repository metadata and an explicit remote-URL allowlist to the content scanner. The default scanner still rejects configured remotes unless the exact intended URL is supplied:

```bash
python scripts/audit_public_release.py --commit HEAD \
  --allow-remote-url https://github.com/TheKingOfFPS/PhysTok.git
```

The original November 8 `PUBLIC_RELEASE_AUDIT.md` and validation receipts describe the pre-upload snapshot. They remain historical evidence. The current public-file manifest is regenerated for this private-upload revision. GitHub visibility and the uploaded commit/tree are separately verified at upload time.

The paper's V12 revision changes author layout/affiliations and appends this repository URL to the abstract. It adds the School of Information Science and Technology affiliation for Dong Han while retaining the Office of Information Management and Planning affiliation. Scientific conclusions, result numbers and model definitions are unchanged. The manuscript PDF/source ZIP are not included in this code repository.

Changing visibility later does not require a new repository URL, provided ownership and repository name are unchanged. The author will decide when to make the repository publicly accessible for submission.
