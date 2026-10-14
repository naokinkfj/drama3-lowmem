# drama3-lowmem-dev plus Nsight Systems CLI for timeline profiling.
FROM drama3-lowmem-dev
RUN apt-get update && (apt-get install -y --no-install-recommends nsight-systems-cli || apt-get install -y --no-install-recommends "nsight-systems-2025*") && rm -rf /var/lib/apt/lists/*
