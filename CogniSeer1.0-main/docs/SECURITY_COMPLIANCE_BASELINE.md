# Security And Compliance Baseline

This project enforces a combined baseline from both selected frameworks.

## Baseline

- OWASP ASVS Level 1 controls for application security posture.
- NIST SSDF-aligned secure development lifecycle practices.
- Software Bill of Materials (SBOM) required for every release artifact set.
- Signed release artifacts required before production deployment.

## Required Security Gates

- Static security scan: `bandit` via `make security`.
- Dependency vulnerability scan: `pip-audit` via `make security`.
- Any ignored CVE must be documented with:
  - vulnerability ID
  - reason for exception
  - expiration date
  - owner

## Current Temporary Exception Register

- `PYSEC-2026-139`
  - reason: no upstream fixed torch release available at update time
  - owner: ML Lead
  - expiration: 2026-09-30
  - mitigation: monitor upstream torch advisory feed and remove ignore immediately when fixed

## Release Blocking Rules

- Any unapproved high/critical finding blocks release.
- Any expired exception blocks release.
- Unsigned artifacts block release.
- Missing SBOM blocks release.
