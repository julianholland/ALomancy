---
description: Check CI, bump version tag, build, and publish to PyPI
allowed-tools: Bash(gh run list:*), Bash(gh run view:*), Bash(git tag:*), Bash(git push:*), Bash(python -m build:*), Bash(twine upload:*), Bash(twine check:*), Bash(rm -rf:*), Bash(pip install:*)
---

## Context

- Latest CI runs on master: !`gh run list --branch master --limit 5 --json conclusion,status,name,headSha,databaseId`
- Current tags: !`git tag --sort=-version:refname | head -5`
- Current version (setuptools_scm): !`python -m setuptools_scm 2>/dev/null || git describe --tags`
- Current branch: !`git branch --show-current`

## Your task

You are publishing a new release of ALomancy to PyPI. Follow these steps in order, stopping and reporting clearly if any step fails.

### Step 1 — Verify CI is green on master

Parse the CI run list above. Find the most recent completed run for the `CI/CD Pipeline` workflow on master. If its conclusion is not `success`, **stop immediately** and tell the user which job failed and what the run URL is (construct it as `https://github.com/julianholland/ALomancy/actions/runs/<databaseId>`). Do not proceed until CI is green.

If CI is still in progress, tell the user and stop.

### Step 2 — Determine the new version tag

The project uses `setuptools_scm` — the version is driven entirely by git tags (format: `vMAJOR.MINOR.PATCH`). Show the user the current latest tag and ask them which version bump they want:
- **patch** (e.g. v0.2.0 → v0.2.1) — bug fixes only
- **minor** (e.g. v0.2.0 → v0.3.0) — new features, backwards-compatible
- **major** (e.g. v0.2.0 → v1.0.0) — breaking changes

Wait for the user to confirm the new tag before proceeding.

### Step 3 — Create and push the git tag

Once the user confirms the new version tag:

```bash
git tag -a <new_tag> -m "Release <new_tag>"
git push origin <new_tag>
```

Confirm the tag was pushed successfully.

### Step 4 — Build the package

Clean any previous build artifacts, then build:

```bash
rm -rf dist/ build/
pip install --quiet build twine
python -m build
twine check dist/*
```

If `twine check` reports any errors, stop and report them. Do not upload a broken package.

### Step 5 — Upload to PyPI

```bash
twine upload dist/*
```

`twine` will prompt for credentials if `~/.pypirc` or the `TWINE_USERNAME`/`TWINE_PASSWORD` environment variables are not set. If you see an auth error, tell the user to either:
- Run `! twine upload dist/*` themselves so they can enter credentials interactively, or
- Set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<their-pypi-api-token>` before invoking this skill again.

### Step 6 — Confirm

Report the published version, the PyPI URL (`https://pypi.org/project/alomancy/<version>/`), and the git tag that was pushed.
