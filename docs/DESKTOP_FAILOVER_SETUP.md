# Linux Desktop Failover Setup

This guide prepares a Linux desktop to serve the AI-Entrepreneurship Construct
Specification Platform when the laptop is unavailable.

## Where the platform runs

The platform does **not** run from the flash drive.

- The flash drive is the private transfer and recovery medium.
- The Git repository is cloned to `~/projects/ETV_V2` on the desktop.
- The analytical runtime archive is extracted into that local repository.
- Credentials and the Cloudflare tunnel token are decrypted into the desktop
  user's private configuration directory.
- The dashboard and Cloudflare connector run from the desktop's internal disk.

The flash drive can be removed after restoration and local verification.

## Current recovery bundle

The verified bundle is:

```text
ETV_DESKTOP_BACKUP/etv-review-host-20260724T181710Z
```

`BUNDLE_INFO` and `CODE_COMMIT` inside the bundle record the exact branch,
commit, source host, and encryption state. Always verify against those files
rather than copying a commit hash from an earlier version of this guide.

Keep the encryption passphrase separate from the flash drive.

## 1. Connect and locate the flash drive

Insert the drive and inspect mounted filesystems:

```bash
lsblk -f
```

Most Linux desktops mount removable media below `/media/$USER`. Locate the
bundle:

```bash
BUNDLE="$(find "/media/$USER" -type d \
  -name 'etv-review-host-20260724T181710Z' \
  -print -quit 2>/dev/null)"

printf 'Bundle: %s\n' "$BUNDLE"
```

If this prints an empty value, identify the mount point from `lsblk -f` and set
the path manually:

```bash
BUNDLE="/actual/mount/path/ETV_DESKTOP_BACKUP/etv-review-host-20260724T181710Z"
```

Verify the bundle before restoring:

```bash
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
```

The expected result is:

```text
./BUNDLE_INFO: OK
./CODE_COMMIT: OK
./host-secrets.tar.gz.enc: OK
./repository.bundle: OK
./runtime.tar: OK
```

Do not continue if any checksum fails.

## 2. Install Linux prerequisites

These commands apply to Ubuntu and Debian:

```bash
sudo apt-get update

sudo apt-get install -y \
  git curl rsync openssl ca-certificates tar
```

## 3. Clone the repository onto the desktop

This command reads the offline repository bundle from the flash drive and
writes a normal Git checkout to the desktop's internal disk:

```bash
mkdir -p ~/projects

git clone -b stage-2a \
  "$BUNDLE/repository.bundle" \
  ~/projects/ETV_V2

cd ~/projects/ETV_V2

EXPECTED_COMMIT="$(tr -d '[:space:]' < "$BUNDLE/CODE_COMMIT")"
CURRENT_COMMIT="$(git rev-parse HEAD)"

printf 'Bundle commit:  %s\n' "$EXPECTED_COMMIT"
printf 'Desktop commit: %s\n' "$CURRENT_COMMIT"

test "$CURRENT_COMMIT" = "$EXPECTED_COMMIT"
```

If `~/projects/ETV_V2` already exists, stop and inspect that checkout instead of
overwriting it:

```bash
git -C ~/projects/ETV_V2 status -sb
git -C ~/projects/ETV_V2 rev-parse HEAD
```

## 4. Install Miniconda and the serving environment

Skip the Miniconda download if `~/miniconda3/bin/conda` already exists.

```bash
curl -fsSL \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/miniconda.sh

bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
```

Create the project environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n graphrag python=3.11 -y

~/miniconda3/envs/graphrag/bin/pip install \
  -r ~/projects/ETV_V2/deploy/review-host-requirements.txt

~/miniconda3/envs/graphrag/bin/pip install \
  -e ~/projects/ETV_V2
```

## 5. Install `cloudflared`

Cloudflare's official Linux downloads are documented at:

<https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/>

Install the current package:

```bash
cd /tmp

curl --location \
  --output cloudflared.deb \
  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$(dpkg --print-architecture).deb"

sudo dpkg -i cloudflared.deb

cloudflared --version
```

## 6. Restore onto the desktop in standby mode

Return to the local checkout:

```bash
cd ~/projects/ETV_V2
```

Restore the analytical data and encrypted host configuration:

```bash
bash scripts/restore_review_host.sh \
  "$BUNDLE" \
  --standby
```

Enter the OpenSSL encryption passphrase when prompted. The restoration:

- extracts `runtime.tar` into `~/projects/ETV_V2`;
- installs `.env` in the local project;
- installs administrator and reviewer credentials in
  `~/.config/etv-dashboard/auth.env`;
- installs the named-tunnel token in
  `~/.config/etv-dashboard/tunnel.token`;
- starts the authenticated dashboard at `127.0.0.1:8321`; and
- keeps the public named tunnel disabled.

## 7. Test the desktop locally

Load the credentials without printing them:

```bash
set -a
source ~/.config/etv-dashboard/auth.env
set +a
```

Test administrator health:

```bash
curl -fsS \
  -u "$ETV_DASHBOARD_USERNAME:$ETV_DASHBOARD_PASSWORD" \
  http://127.0.0.1:8321/api/health
```

Test the reviewer restriction:

```bash
curl -fsS \
  -u "$ETV_DASHBOARD_REVIEW_USERNAME:$ETV_DASHBOARD_REVIEW_PASSWORD" \
  http://127.0.0.1:8321/api/access-mode
```

The reviewer response must include:

```json
{
  "role": "reviewer",
  "read_only": true,
  "writes_allowed": false
}
```

Open the local platform in the desktop browser:

```text
http://127.0.0.1:8321
```

Run the host status check:

```bash
cd ~/projects/ETV_V2
bash scripts/review_host_status.sh
```

In standby mode, `Named tunnel active` should be `PENDING`. That is intentional;
all other checks should be `OK`.

## 8. Enable unattended startup

Allow the user services to start at boot without an interactive login:

```bash
sudo loginctl enable-linger "$USER"
```

In the desktop's power settings:

- disable automatic sleep while connected to power;
- restore power automatically after an outage if the firmware provides that
  option; and
- keep the network connection enabled during unattended operation.

Do not activate the desktop's public tunnel until local testing succeeds.

## 9. Transfer the public site from laptop to desktop

Only one unsynchronised host should publish the named tunnel.

First, on the laptop:

```bash
systemctl --user disable --now etv-dashboard-tunnel-named.service
```

Then, on the desktop:

```bash
cd ~/projects/ETV_V2

bash scripts/install_review_host.sh
bash scripts/review_host_status.sh
bash scripts/smoke_review_site.sh https://aitheoryelaboration.org
```

The permanent review URL remains:

```text
https://aitheoryelaboration.org
```

## 10. After cutover

The desktop becomes the active writable host. After any administrator annotation
or other write:

1. treat the desktop databases as the current authoritative state;
2. create a new encrypted failover bundle before switching back to the laptop;
3. do not activate an older unsynchronised copy; and
4. keep credentials, the tunnel token, and the encryption passphrase outside
   Git.

After the desktop has passed local and public testing, the flash drive may be
unmounted and removed:

```bash
sync
```

Use the desktop file manager's eject control before physically removing it.
