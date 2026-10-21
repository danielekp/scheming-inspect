FROM aisiuk/inspect-tool-support@sha256:bbaecf7a093cfe24703c97d562bb059899e272cd5d57f16d9916fe05e6614419
RUN mkdir -p /workspace && chown 1000:1000 /workspace
USER 1000:1000
WORKDIR /workspace
