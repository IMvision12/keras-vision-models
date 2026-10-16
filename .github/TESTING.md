# Pull request test commands

Lint & Format continues to run automatically on pull requests. Model, tokenizer,
image-processor, and processor tests run on a pull request only when **IMvision12**
posts a new conversation comment containing exactly:

```text
/run-tests
```

Other users' commands, issue comments, edited comments, and other text do not
authorize the test suite. Re-running a comment-triggered test job also requires
IMvision12 to be the triggering account.

The command starts the existing 16 test jobs: 9 model shards (Torch, TensorFlow,
and JAX), 1 tokenizer job, 3 image-processor jobs, and 3 processor jobs. They test
the open PR's head commit captured when the request is authorized. A later push
does not start another test run; post `/run-tests` again to test the new commit.

Results appear on the tested PR commit as **Requested model, tokenizer & processor
tests**, with a link to the workflow's individual jobs. If these tests must block
merging, require this aggregate check in the repository's branch protection or
ruleset; automatically skipped test jobs are not a substitute for that check.

The updated workflow must be present on the repository's default branch before
GitHub delivers comment-triggered runs. No additional personal access token or
repository secret is needed.

Release-triggered tests still run automatically after lint succeeds. The release,
documentation, and standalone cross-backend parity workflows are unchanged.

Authorization and result-reporting jobs have check-writing permission but never
check out PR code. Test jobs have read-only repository access and do not persist
checkout credentials. Comment-triggered tests do not use shared Actions caches
or publish Codecov coverage from the default-branch event context; release runs
retain their existing cache and coverage steps.
