# Security Review

This artifact repository was scanned before publication for embedded credentials and local-machine leakage. The repository is intended to contain paper artifacts, curated result data, and an installable wrapper package only.

## Checks Performed

- No `.env`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa*`, `known_hosts`, credential, token, or secret-named files were present.
- No executable files were present. Python scripts are source files only.
- No private-key blocks, AWS access-key IDs, OpenAI `sk-` keys, GitHub PATs, Slack tokens, Google API keys, Azure storage connection strings, SAS markers, or JWT-shaped credentials were found in text or binary files.
- No absolute local source-machine paths remain in the repository content.
- The only real-looking email intentionally present is the paper author contact address in `paper/think_before_you_speak_v2.tex`; other emails in datasets are synthetic examples such as `example.com` prompts/results.
- Documentation mentions environment variable names such as `OPENAI_API_KEY` and `AZURE_OPENAI_KEY`, but only as placeholders or code references; no values are included.

## Re-run Locally

```bash
python scripts/validate_artifact.py
shasum -a 256 -c checksums.sha256
```

The validator includes strict credential-signature checks for common key formats and should be run before publishing any modified version of this repository.

Note: `scripts/validate_artifact.py` ignores the local `.git/` metadata directory so the check can run inside a normal Git clone. It still checks the repository content that would be committed and published.
