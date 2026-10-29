FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
# Debian already includes Bash and coreutils; the scenarios only need Python's stdlib.
RUN groupadd --gid 1000 worker \
    && useradd --uid 1000 --gid worker --create-home --shell /bin/bash worker \
    && mkdir -p /workspace && chown 1000:1000 /workspace
USER 1000:1000
WORKDIR /workspace
