Kestrel/Moondream review evidence

attention-policy: model-free reproductions, numerical sweeps, selector patch and caller verification.
vision-batching: portable probes and SDK throughput/quality measurements.
nvfp4-vision: integrated-runtime probe and SDK throughput/quality measurements.

Patches use git apply or patch -p1, not git am. NVFP4 is stacked on batching commit 6037cc89d0e237654a6dc1b91e28dc533cd1bfb6. Attention callers need a compatible kernel release before merge.

No model weights or input images are included. Model-free attention reproductions need no images. Model probes accept supplied weights and representative images; exact frozen-panel metrics cannot be recreated from this bundle alone. Sequence identifiers are anonymized consistently. Operational metadata is omitted; numerical results and regressions are retained. Historical source hashes refer to the tested code; the publication changes NVFP4 README text only. The independent kernel patch is included here because its upstream repository has not been established.
