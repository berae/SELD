#!/usr/bin/env bash
set -u -o pipefail

host_name=$(hostname)
if [[ "$host_name" == rabbit02* ]]; then
  project_root=/work/zhanghc/Myllm/SELD/ObjectStateSELD
else
  project_root=/root/autodl-tmp/SELD/ObjectStateSELD
fi
download_root="$project_root/data_raw/p1_downloads"
log_root="$project_root/logs"
state_file="$log_root/p1_official_download_20260807.state"

cd "$project_root" || exit 72
mkdir -p "$download_root" "$log_root"

# rabbit02 can reach Zenodo directly. AutoDL requires the configured academic
# proxy in the current network environment, even though Zenodo is slower there.
if [[ "$host_name" == rabbit02* ]]; then
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
else
  source /etc/network_turbo
fi

download_one() {
  local dataset="$1"
  local filename="$2"
  local expected_bytes="$3"
  local expected_md5="$4"
  local url="$5"
  local dest_dir="$download_root/$dataset"
  local final_file="$dest_dir/$filename"
  local partial_file="$dest_dir/$filename.part"
  local actual_bytes actual_md5 code

  mkdir -p "$dest_dir"

  if [[ -s "$final_file" ]]; then
    actual_bytes=$(stat -c '%s' "$final_file")
    actual_md5=$(md5sum "$final_file" | awk '{print $1}')
    if [[ "$actual_bytes" == "$expected_bytes" && "$actual_md5" == "$expected_md5" ]]; then
      echo "[$(date --iso-8601=seconds)] SKIP_VERIFIED $dataset/$filename"
      return 0
    fi
    echo "Existing final file failed verification: $final_file" >&2
    return 74
  fi

  printf 'RUNNING %s/%s %s\n' "$dataset" "$filename" "$(date --iso-8601=seconds)" > "$state_file"
  echo "[$(date --iso-8601=seconds)] START $dataset/$filename expected_bytes=$expected_bytes"

  aria2c \
    --continue=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=16M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --max-tries=3 \
    --retry-wait=5 \
    --timeout=60 \
    --connect-timeout=30 \
    --dir="$dest_dir" \
    --out="$filename.part" \
    "$url"
  code=$?
  if [[ "$code" -ne 0 ]]; then
    printf 'FAILED %s/%s %s exit=%s\n' "$dataset" "$filename" "$(date --iso-8601=seconds)" "$code" > "$state_file"
    return "$code"
  fi

  actual_bytes=$(stat -c '%s' "$partial_file")
  actual_md5=$(md5sum "$partial_file" | awk '{print $1}')
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "Size mismatch for $dataset/$filename: expected=$expected_bytes actual=$actual_bytes" >&2
    printf 'FAILED %s/%s %s size_mismatch\n' "$dataset" "$filename" "$(date --iso-8601=seconds)" > "$state_file"
    return 75
  fi
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    echo "MD5 mismatch for $dataset/$filename: expected=$expected_md5 actual=$actual_md5" >&2
    printf 'FAILED %s/%s %s md5_mismatch\n' "$dataset" "$filename" "$(date --iso-8601=seconds)" > "$state_file"
    return 76
  fi

  mv "$partial_file" "$final_file"
  printf '%s  %s\n' "$expected_md5" "$filename" >> "$dest_dir/MD5SUMS.txt"
  sync
  echo "[$(date --iso-8601=seconds)] PASS $dataset/$filename bytes=$actual_bytes md5=$actual_md5"
}

einv2_dir="$project_root/external/checkpoints/einv2_zenodo_4158864"
einv2_final="$einv2_dir/out_train.zip"
einv2_partial="$einv2_dir/out_train.zip.part"
einv2_bytes=385862385
einv2_md5=7bb727438e74c726eb5393ac4b142247
mkdir -p "$einv2_dir"
printf 'RUNNING einv2_checkpoint %s\n' "$(date --iso-8601=seconds)" > "$state_file"
echo "[$(date --iso-8601=seconds)] RESUME EINV2 checkpoint from official Zenodo source host=$host_name"
if [[ ! -s "$einv2_final" ]]; then
  aria2c \
    --continue=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=16M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --max-tries=3 \
    --retry-wait=5 \
    --timeout=60 \
    --connect-timeout=30 \
    --dir="$einv2_dir" \
    --out='out_train.zip.part' \
    'https://zenodo.org/records/4158864/files/out_train.zip?download=1' || {
      code=$?
      printf 'FAILED einv2_checkpoint %s exit=%s\n' "$(date --iso-8601=seconds)" "$code" > "$state_file"
      exit "$code"
    }
  actual_bytes=$(stat -c '%s' "$einv2_partial")
  actual_md5=$(md5sum "$einv2_partial" | awk '{print $1}')
  if [[ "$actual_bytes" != "$einv2_bytes" || "$actual_md5" != "$einv2_md5" ]]; then
    echo "EINV2 verification failed: bytes=$actual_bytes md5=$actual_md5" >&2
    printf 'FAILED einv2_checkpoint %s verification\n' "$(date --iso-8601=seconds)" > "$state_file"
    exit 75
  fi
  mv "$einv2_partial" "$einv2_final"
  rm -f "$einv2_dir/out_train.zip.part.aria2"
  printf '%s  %s\n' "$einv2_md5" 'out_train.zip' > "$einv2_dir/MD5SUMS.txt"
fi
echo "[$(date --iso-8601=seconds)] PASS EINV2 checkpoint"

download_one starss22 'foa_dev.zip' 2199498647 165dd033b262dc11a8853635c1def59b \
  'https://zenodo.org/api/records/6387880/files/foa_dev.zip/content' || exit $?
download_one starss22 'metadata_dev.zip' 634378 b460e17e0848c49f03f238afb89fa87e \
  'https://zenodo.org/api/records/6387880/files/metadata_dev.zip/content' || exit $?
download_one starss22 'README.md' 23568 0e12de6f61c43e3d6af05911b46c6663 \
  'https://zenodo.org/api/records/6387880/files/README.md/content' || exit $?
download_one starss22 'LICENSE' 1194 2424296dab0b421211874b6c5cd1cacb \
  'https://zenodo.org/api/records/6387880/files/LICENSE/content' || exit $?

download_one starss23 'foa_dev.zip' 3406085425 316b834ee6393c22862c314cc3c7ebb0 \
  'https://zenodo.org/api/records/7880637/files/foa_dev.zip/content' || exit $?
download_one starss23 'foa_eval.zip' 1643937003 7caa61bb6ea997f5bfac5b20177f505a \
  'https://zenodo.org/api/records/7880637/files/foa_eval.zip/content' || exit $?
download_one starss23 'metadata_dev.zip' 1165845 e73af95a6d5f3f7e009ac6a70804f44a \
  'https://zenodo.org/api/records/7880637/files/metadata_dev.zip/content' || exit $?
download_one starss23 'README.md' 28707 bbfde2b9d0e47c7dafd3cbc92107c920 \
  'https://zenodo.org/api/records/7880637/files/README.md/content' || exit $?
download_one starss23 'LICENSE' 1194 1c11108eda7c915172b10c48276cc189 \
  'https://zenodo.org/api/records/7880637/files/LICENSE/content' || exit $?

download_one tau_nigens_2021 'foa_dev.z01' 4294967296 270a94dc5cd183ea6532c5a3f6e9036c \
  'https://zenodo.org/api/records/5476980/files/foa_dev.z01/content' || exit $?
download_one tau_nigens_2021 'foa_dev.zip' 1425535805 80648b5f64b1b4a824084560f1334f54 \
  'https://zenodo.org/api/records/5476980/files/foa_dev.zip/content' || exit $?
download_one tau_nigens_2021 'foa_eval.zip' 1893873528 591f8d2b500a671ae34822b4ff1e0889 \
  'https://zenodo.org/api/records/5476980/files/foa_eval.zip/content' || exit $?
download_one tau_nigens_2021 'metadata_dev.zip' 1930087 cd8cd8b4dc9a3e3df91ac55c1ccf73b7 \
  'https://zenodo.org/api/records/5476980/files/metadata_dev.zip/content' || exit $?
download_one tau_nigens_2021 'metadata_eval.zip' 660393 11c021253c8b55fd74083bd0a35c2ee4 \
  'https://zenodo.org/api/records/5476980/files/metadata_eval.zip/content' || exit $?
download_one tau_nigens_2021 'README.md' 22689 73332d04f05ed6568210731fb9296593 \
  'https://zenodo.org/api/records/5476980/files/README.md/content' || exit $?

download_one dcase2024_synthetic 'DCASE Task 3 synthetic dataset 2024.zip.001' 4697620480 c651f4a9670326097361763015d955f1 \
  'https://zenodo.org/api/records/10932241/files/DCASE%20Task%203%20synthetic%20dataset%202024.zip.001/content' || exit $?
download_one dcase2024_synthetic 'DCASE Task 3 synthetic dataset 2024.zip.002' 4697620480 3199250c0c7fa718888f74a7d3325400 \
  'https://zenodo.org/api/records/10932241/files/DCASE%20Task%203%20synthetic%20dataset%202024.zip.002/content' || exit $?
download_one dcase2024_synthetic 'DCASE Task 3 synthetic dataset 2024.zip.003' 4697620480 203df900f0275d8c34f04fa508b60a85 \
  'https://zenodo.org/api/records/10932241/files/DCASE%20Task%203%20synthetic%20dataset%202024.zip.003/content' || exit $?
download_one dcase2024_synthetic 'DCASE Task 3 synthetic dataset 2024.zip.004' 4697620480 12306d0d398327847734896eae3986c1 \
  'https://zenodo.org/api/records/10932241/files/DCASE%20Task%203%20synthetic%20dataset%202024.zip.004/content' || exit $?
download_one dcase2024_synthetic 'DCASE Task 3 synthetic dataset 2024.zip.005' 140056763 d1e53c4c21dab44a1a6a6c150bc90e3f \
  'https://zenodo.org/api/records/10932241/files/DCASE%20Task%203%20synthetic%20dataset%202024.zip.005/content' || exit $?

printf 'COMPLETED %s\n' "$(date --iso-8601=seconds)" > "$state_file"
sync
echo "[$(date --iso-8601=seconds)] ALL P1 FOA DOWNLOADS COMPLETED"
