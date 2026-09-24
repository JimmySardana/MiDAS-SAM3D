# Fraunhofer cluster deployment

This guide follows the repository's current `scripts/` configuration. The registry host, login host, username, NAS location, SSH key, and ports are set in those scripts; review them for your account and cluster before running the commands.

## 1. Prepare and upload the SAM 3D Objects checkpoints

Request access to the [official SAM 3D Objects checkpoints](https://huggingface.co/facebook/sam-3d-objects), then download them as described in the [main README](../README.md#get-the-checkpoints). The Slurm script expects a ZIP archive at:

```text
smb://pc3163/nobackup/spool/sam3d_checkpoint.zip
```

It extracts that archive into `/opt/cache/ldapUser/sam3d_checkpoint` on the compute node and mounts the extracted directory as `/checkpoints` in the container. Therefore, `pipeline.yaml` must be at the **root of the ZIP archive**, not inside an additional `hf/` or `checkpoints/` directory.

On a Linux host with `zip` and access to the NAS, create the archive from the contents of `checkpoints/hf`:

```bash
cd checkpoints/hf
zip -r ../../sam3d_checkpoint.zip .
cd ../..
unzip -Z1 sam3d_checkpoint.zip | grep -E '^(\./)?pipeline\.yaml$'
```

The final command should print `pipeline.yaml` (possibly prefixed with `./`). Upload the archive to the exact NAS path used by `scripts/sam3d_sbatch.sh`:

```bash
gio mount smb://pc3163/nobackup
gio copy --overwrite sam3d_checkpoint.zip smb://pc3163/nobackup/spool/sam3d_checkpoint.zip
gio info smb://pc3163/nobackup/spool/sam3d_checkpoint.zip
```

Authenticate to the NAS when prompted. The Slurm script mounts the same share on a compute node using `~/smbcreds`, so that credential file must be available to the job account. It copies the archive only if `/opt/cache/ldapUser/sam3d_checkpoint` does not already exist. If you update the archive later, also refresh that cached directory before restarting the job.

## 2. Build and push the image

From the repository root on a machine that can reach `pc3163.igd.fraunhofer.de:4567` and run Docker:

```bash
bash scripts/build.sh
bash scripts/build_push.sh
```

`build.sh` tags the image as `pc3163.igd.fraunhofer.de:4567/sam3d`. `build_push.sh` logs in to that registry and pushes the tag. The model checkpoints are not part of the image.

## 3. Review and submit the Slurm job

Check the user settings near the top of `scripts/sam3d_sbatch.sh`, especially `LOGIN_HOST`, `LOGIN_USER`, `SSH_KEY`, `DATAPATH`, `CONFIG_PATH`, and the ports. The current script requests one GPU, 64 CPUs per GPU, 128 GB of memory, and a one-hour job. It needs `rootless-docker`, `gio`, `unzip`, SSH access to the login host using the specified key, and registry access on the compute node.

From a checkout available to the cluster:

```bash
sbatch scripts/sam3d_sbatch.sh
squeue -u "$USER"
```

The job copies and extracts the NAS archive if necessary, then runs the image with the checkpoint directory mounted at `/checkpoints` and `SAM3D_CONFIG=/checkpoints/pipeline.yaml`. The container listens on port `8000`; the job binds it to `127.0.0.1:8000` on the compute node and forwards login-host port `18000` back to that address. The job's Slurm output and error files are named `slurm-sam3d-cluster-<job-id>.out` and `.err`.

## 4. Access the API

On the configured login host, verify the service through the reverse tunnel:

```bash
curl http://127.0.0.1:18000/health
```

To call it from your workstation, create a local SSH forward in a separate terminal (replace the host and username if you changed the script):

```bash
ssh -N -L 8001:127.0.0.1:18000 ldapUser@amperecontrol1
```

The service is then reachable at `http://localhost:8001`. For example:

```bash
curl --fail --show-error \
  -F "image=@test/inference/images/image.png" \
  -F "mask=@test/inference/images/mask.png" \
  http://localhost:8001/infer/glb \
  -o result.glb
```

The image and mask must have the same dimensions, and the mask's alpha channel must be transparent outside the target object. See the [API instructions](../README.md#generate-a-glb) for the response format.

## Common issues

- **Missing `pipeline.yaml` at startup:** inspect the ZIP layout and confirm the extracted file is `/opt/cache/ldapUser/sam3d_checkpoint/pipeline.yaml`.
- **Updated archive has no effect:** the script skips the NAS copy whenever the cache directory already exists. Refresh the cache before launching a new job.
- **Health check cannot connect:** inspect the Slurm output and error files for container startup, NAS authentication, or SSH tunnel errors. Confirm that login-host port `18000` is available.
