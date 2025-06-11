# Docker Notes

RDKit requires system libraries when building from source on ARM systems.
Install `libboost-all-dev`, `cmake`, and standard build tools before
installing the Python package.

For environments without a local sandbox, pull the image from GHCR:

```bash
docker run --rm -it ghcr.io/<owner>/tsce_sandbox:latest
```
