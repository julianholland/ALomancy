---
description: Check CI, bump version tag, build, and publish to PyPI
allowed-tools: Bash(gh run list:*), Bash(gh run view:*), Bash(git status:*), Bash(git add:*), Bash(git commit:*), Bash(git tag:*), Bash(git describe:*), Bash(git push:*), Bash(python -m build:*), Bash(twine upload:*), Bash(twine check:*), Bash(rm -rf:*), Bash(pip install:*)
---

## Context

- Latest CI runs on master: !`gh run list --branch master --limit 5 --json conclusion,status,name,headSha,databaseId`
- Current tags: !`git tag --sort=-version:refname | head -5`
- Current branch: !`git branch --show-current`
- Uncommitted changes: !`git status --short`

## Your task

You are publishing a new release of ALomancy to PyPI. Follow these steps in order, stopping and reporting clearly if any step fails.

### Step 1 — Verify CI is green on master

Parse the CI run list above. Find the most recent completed run for the `CI/CD Pipeline` workflow on master. If its conclusion is not `success`, **stop immediately** and tell the user which job failed and what the run URL is (construct it as `https://github.com/julianholland/ALomancy/actions/runs/<databaseId>`). Do not proceed until CI is green.

If CI is still in progress, tell the user and stop.

### Step 2 — Ensure a clean working tree

Check the `Uncommitted changes` output above. If any tracked files are modified or staged, **commit them before tagging**:

```bash
git add <files>
git commit -m "chore: pre-release cleanup"
```

An uncommitted file causes `setuptools_scm` to append `.post0` to the version (e.g. `0.3.0.post0` instead of `0.3.0`). The tag must sit on a clean commit.

Only proceed to Step 3 once `git status --short` shows no modified tracked files. Untracked files (lines beginning with `??`) are fine and can be ignored.

### Step 3 — Determine the new version tag

The project uses `setuptools_scm` — the version is driven entirely by git tags (format: `vMAJOR.MINOR.PATCH`). Show the user the current latest tag and ask them which version bump they want:
- **patch** (e.g. v0.2.0 → v0.2.1) — bug fixes only
- **minor** (e.g. v0.2.0 → v0.3.0) — new features, backwards-compatible
- **major** (e.g. v0.2.0 → v1.0.0) — breaking changes

Wait for the user to confirm the new tag before proceeding.

### Step 4 — Create and push the git tag

Once the user confirms the new version tag:

```bash
git tag -a <new_tag> -m "Release <new_tag>"
git push origin <new_tag>
```

Confirm the tag was pushed successfully.

### Step 5 — Build the package

Clean any previous build artifacts, then build:

```bash
rm -rf dist/ build/
pip install --quiet build twine
python -m build
twine check dist/*
```

Verify the built version string (shown in the `python -m build` output) is exactly `<new_tag>` without any `.post0` or `.devN` suffix. If it has a suffix, **stop**: there are uncommitted changes or extra commits since the tag — go back to Step 2.

If `twine check` reports any errors, stop and report them. Do not upload a broken package.

### Step 6 — Upload to PyPI

```bash
twine upload dist/*
```

`twine` requires credentials. It reads `~/.pypirc` automatically if present. The recommended setup:

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-<your-token-here>
```

If credentials are not configured and you see an auth error or `EOFError` (twine tried to prompt but can't in a non-interactive terminal), tell the user to:
- Create `~/.pypirc` as above, then run `! twine upload dist/*` themselves, **or**
- Run `! TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<token> twine upload dist/*` to pass the token inline.

### Step 7 — Confirm

Report the published version, the PyPI URL (`https://pypi.org/project/alomancy/<version>/`), and the git tag that was pushed.
