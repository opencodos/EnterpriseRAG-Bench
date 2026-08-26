#!/usr/bin/env bash
# Create the baseline host: the box that serves the study's reader and runs both
# reproduced arms.
#
# **This is not the measurement instance and must never become it.** The Aethos arm is
# timed single-stream on an idle client-spec deployment; an index build or an agent
# sweep running beside it would be measuring the neighbour. The two live on separate
# hosts for that reason, and the consequence — accuracy comparable across arms, latency
# not — is a disclosed property of the results rather than an accident.
#
# Spot, because both arms resume by question id: a preemption costs the question in
# flight and nothing else. It does not touch accuracy, and baseline latency is already
# reported per host rather than against the Aethos arm.
set -euo pipefail

PROJECT="${PROJECT:-aethos-prod}"
ZONE="${ZONE:-us-east1-b}"
NAME="${NAME:-erb-baselines}"
# 2x A100 40GB. The project has no A100-80GB quota, and a 27B at bf16 is ~54GB of
# weights, so the reader is tensor-parallel across two cards rather than resident on one.
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-2g}"
# Weights (~54GB), the HF cache, a materialized tier tree (~160MB at T13) and the
# OpenSearch index. The corpus itself never lands here — only tiers do.
BOOT_DISK_GB="${BOOT_DISK_GB:-500}"
# A Deep Learning VM image, because the NVIDIA driver is preinstalled on it. A stock
# Ubuntu image means building the driver at boot, and that build fails against recent
# GCP kernels -- measured here on 6.17.0-1022-gcp, which cannot compile it. A host that
# comes up or does not depending on which kernel the image last shipped is not one the
# ladder can be re-measured on months from now.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2404-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud compute instances create "${NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size="${BOOT_DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script="${here}/startup.sh" \
  --scopes=cloud-platform \
  --labels=purpose=erb-baselines,programme=erb-ladder-v1

cat <<EOF

Created ${NAME} in ${ZONE}. The startup script installs the driver and pulls the
weights, which takes a while. Watch it with:

  gcloud compute ssh ${NAME} --zone ${ZONE} --project ${PROJECT} \\
    --command 'sudo journalctl -u google-startup-scripts -f'

Then, from a session on the box:

  curl -s localhost:8000/v1/models | head
  curl -s localhost:9200 | head

Reach it from a workstation over an SSH tunnel rather than a firewall rule — the reader
has no authentication of its own:

  gcloud compute ssh ${NAME} --zone ${ZONE} --project ${PROJECT} -- -N \\
    -L 8000:localhost:8000 -L 9200:localhost:9200
EOF
