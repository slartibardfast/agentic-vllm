# Environment matrix (the two rigs)

The two rigs run an identical software stack so that a measured difference
reflects the hardware and not the toolchain. Aligned on 2026-08-27.

| component | this host | rope (192.168.178.60) |
|---|---|---|
| OS | Arch Linux | Arch Linux |
| kernel package | `linux-lts 6.18.47-1` | `linux-lts 6.18.47-1` |
| running kernel | `6.18.46-1-lts` (reboot pending) | `6.18.47-1-lts` |
| NVIDIA kernel module | `nvidia-open-dkms 610.57.04-1` | `nvidia-open-dkms 610.57.04-1` |
| NVIDIA userland | `nvidia-utils 610.57.04-1` | `nvidia-utils 610.57.04-1` |
| OpenCL | `opencl-nvidia 610.57.04-1` | `opencl-nvidia 610.57.04-1` |
| CUDA toolkit | `cuda 13.3.1-1` (`/opt/cuda`, nvcc 13.3) | `cuda 13.3.1-1` (`/opt/cuda`, nvcc 13.3) |
| GPU | 2× Quadro RTX 6000 (TU102, sm_75, 24 GB) | RTX 3060 Ti (sm_86, 8 GB) |
| graphics clock lock | 1455 MHz (both cards) | 1665 MHz |
| power limit | 260 W | 220 W |

## Notes for anyone running measurements here

- Clock locks are driver state and they reset at every boot. Re-apply after
  each boot, on either rig, before trusting a number:

  ```sh
  sudo nvidia-smi -pm 1
  sudo nvidia-smi -lgc 1455,1455   # rope: 1665,1665
  ```

- rope boots with systemd-boot, default entry `arch-linux-lts.conf` (created
  2026-08-27; kernel options copied from the previous mainline boot). The
  default is persisted twice: as the `default` line in
  `/boot/loader/loader.conf` and in the firmware's
  `LoaderEntryDefault` variable, so a firmware reset cannot silently fall
  back to the mainline kernel. The mainline UKI (`arch-linux.efi`) stays
  installed as a fallback, and its DKMS module is kept at the same driver
  version, so the fallback still drives the GPU if the LTS boot ever fails.
- This host boots systemd-boot with `default arch-lts.conf`, which points at
  the stable filenames `vmlinuz-linux-lts` and `initramfs-linux-lts.img`; the
  package upgrade needed no loader change, and the next reboot picks up the
  newer point release.
- This host runs the previous LTS point release (`6.18.46-1-lts`) until its
  next reboot; the installed package is already `6.18.47-1`. The driver and
  the CUDA toolkit are userspace components and are identical today, which is
  what the benchmark lanes depend on.
- The alignment procedure applied on rope, for the record: `pacman -Sy`,
  then `pacman -S linux-lts linux-lts-headers`, then swap the prebuilt
  module for the DKMS one (`pacman -R nvidia-open`,
  `pacman -S nvidia-open-dkms opencl-nvidia`), then `pacman -S cuda`.
