# Pull request checks

## Automatic, on every pull request

Every pull request (from anyone) runs, with no command needed:

- **Tests / Lint & Format** (`test_code.yml`): `ruff check` and `ruff format --check`.
- **Docs / build** (`docs.yml`): the strict MkDocs build and the asset check. Deploying
  the site stays `main`-only.

## Model, tokenizer and processor tests, on request

The heavy suite (`model_tests.yml`: 9 model shards across Torch, TensorFlow and JAX,
1 tokenizer job, 3 image-processor jobs, 3 processor jobs, and one summary job) runs on
a pull request only when **IMvision12** posts a new comment containing exactly:

```text
/run-tests
```

Other users' comments, edited comments and other text are ignored.

How it works: `run_tests_command.yml` (triggered by the comment, never checking out PR
code) validates the request and dispatches `model_tests.yml` **on the pull request's
branch**, pinned to the PR head commit at that moment. A dispatched run is attached to
that commit, so every test job appears in the PR's checks list as
**Model tests / ...**, alongside the automatic checks. The comment gets a 🚀 reaction and
a reply with the tested commit.

- A later push is not tested automatically: comment `/run-tests` again. A new request on
  the same PR cancels the previous run.
- The branch must contain the current `model_tests.yml` (with its `workflow_dispatch`
  trigger); otherwise the reply asks to merge or rebase `main`.
- **Pull requests from forks**: a fork's branch cannot be dispatched, so the tests run
  from `main` on the PR head commit. They are not attached to the PR; the reply links
  the run.
- `run_tests_command.yml` only takes effect once it is on `main` (comment events always
  use the default branch's workflow files).

**Model tests / Requested model, tokenizer & processor tests** passes only when every
test job passes. To make the suite block merging, require that check in the branch
protection rules or ruleset.

## Releases

Release events run lint and the whole suite in `model_tests.yml` automatically, with
pip caching and Codecov upload (both disabled for pull request runs).

## Security

The comment workflow has `actions: write` (to dispatch) and comment permissions, and never
checks out or executes PR code. The test jobs that run PR code have read-only repository
access, do not persist checkout credentials, and use no secrets or shared caches.
