"""Fetches the data and models ilc_tcga.ipynb needs, only what is missing locally:

- UNI2-h weights (models/uni2-h/, for LRP) and its TCGA-BRCA tile features (data/uni2/h5/, from 60 GB of archives)
- CONCH weights, to the Hugging Face cache, where conch looks for them
- Kather tumour outlines and the Thennavan review tables (data/external/), md5-checked

UNI2-h and CONCH are gated on Hugging Face: request access first. Their weights and the features must not be
redistributed, so they stay git-ignored. The GDC metadata and labels come from the notebook's first step.
"""
import hashlib
import io
import os
import tarfile
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download, login

root = os.path.dirname(os.path.realpath(__file__))
model_repo = "MahmoodLab/UNI2-h"
features_repo = "MahmoodLab/UNI2-h-features"
conch_repo = "MahmoodLab/conch"
model_dir = os.path.join(root, "models/uni2-h")
features_dir = os.path.join(root, "data/uni2")
archives = ["TCGA/TCGA-BRCA_IDC.tar.gz", "TCGA/TCGA-BRCA_OTHERS.tar.gz"]
external_dir = os.path.join(root, "data/external")
# md5 as published in the Zenodo record
kather_url = "https://zenodo.org/api/records/5320076/files"
kather_files = {"BRCA.zip": "3657401cd3341c9f32d8a83b97b1c63f"}
thennavan_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9028992/supplementaryFiles"
# md5 of the files as fetched on 2026-11-04; the supplement has no published checksum
thennavan_files = {"mmc2.xlsx": "fa582c98b15f842a5993104c31a03e8f",
                   "mmc3.xlsx": "80f73527dac9000e5d76e67af4684677"}


def verify(path, md5):
    # path: local file to check
    # md5: expected md5 hex digest of that file
    # hash in chunks, the archives are large
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != md5:
        raise ValueError(f"md5 mismatch for {path}: got {h.hexdigest()}, expected {md5}")


def fetch_features(features_dir):
    h5_dir = os.path.join(features_dir, "h5")
    os.makedirs(h5_dir, exist_ok=True)

    for archive in archives:
        # written only after a complete extraction, so a half-extracted archive is redone
        done = os.path.join(h5_dir, os.path.basename(archive) + ".done")
        if os.path.isfile(done):
            continue

        # each archive holds one flat .h5 per slide, named after the GDC file_name
        local = os.path.join(features_dir, archive)
        if not os.path.isfile(local):
            hf_hub_download(features_repo, archive, repo_type="dataset", local_dir=features_dir)
        with tarfile.open(local, "r:gz") as tar:
            tar.extractall(h5_dir, filter="data")
        open(done, "w").close()


def fetch_zenodo(record_url, files, folder):
    # record_url: Zenodo API url of the record's files
    # files: file name -> md5 published in the record
    # folder: folder that holds (or will receive) the files
    os.makedirs(folder, exist_ok=True)
    # written to .incomplete and renamed, so a present file is a finished download
    for name, md5 in files.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(f"{record_url}/{name}/content", path + ".incomplete")
            os.rename(path + ".incomplete", path)
        verify(path, md5)


def fetch_thennavan(thennavan_dir):
    # thennavan_dir: folder that holds (or will receive) mmc2.xlsx and mmc3.xlsx
    os.makedirs(thennavan_dir, exist_ok=True)
    paths = {name: os.path.join(thennavan_dir, name) for name in thennavan_files}
    if not all(os.path.isfile(p) for p in paths.values()):
        # the supplement zip (19 MB, all figures and tables) is read in memory, only the two tables are kept
        with urllib.request.urlopen(thennavan_url) as resp:
            supplement = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name, path in paths.items():
            with open(path + ".incomplete", "wb") as f:
                f.write(supplement.read(name))
            os.rename(path + ".incomplete", path)
    for name, path in paths.items():
        verify(path, thennavan_files[name])


def main():
    # no parameters: fetches into the paths set at the top of download.py
    # no prompt if a token is stored
    login()
    if not os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")):
        hf_hub_download(model_repo, filename="pytorch_model.bin", local_dir=model_dir)
    # the two files conch's create_model_from_pretrained downloads, into the Hugging Face cache where it looks for them
    for name in ["meta.yaml", "pytorch_model.bin"]:
        hf_hub_download(conch_repo, filename=name)
    fetch_features(features_dir)
    fetch_zenodo(kather_url, kather_files, os.path.join(external_dir, "kather2021"))
    fetch_thennavan(os.path.join(external_dir, "thennavan2021"))
    print(f"UNI2-h in {model_dir}, CONCH in the Hugging Face cache, features in {features_dir}/h5, outlines and review "
          f"tables in {external_dir}")


if __name__ == "__main__":
    main()
