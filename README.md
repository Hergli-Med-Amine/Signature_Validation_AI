# Signature Validation AI

Offline signature verification: a HOG + LBP baseline versus a ResNet-18 embedder trained to recognise people, compared using equal error rate (EER) on skilled and random forgeries.

## Project structure

```
Signature_Validation.ipynb   main notebook (reads everything relative to this folder)
scripts/rebuild_data.py      rebuilds data/raw and outputs/ from archives/
requirements.txt

archives/                    the dataset as 20 MB 7z volumes (GitHub's file limit is 100 MB)
  data/     data-*.7z.001 … .055      images + manifest (1.15 GB)
  outputs/  outputs-*.7z.001 … .003   split.csv + trained model

data/
  manifest.csv               one row per image: path, person_id, writer_id, source, label, md5
  raw/                       (not in git) genuine/ and forged/ images, 42,374 files, 4.6 GB
outputs/
  split.csv                  person → train / val / test
  resnet18_people.pt         trained ResNet-18 weights
```

## Getting started

```bash
git clone https://github.com/Hergli-Med-Amine/Signature_Validation_AI.git
cd Signature_Validation_AI
pip install -r requirements.txt
python scripts/rebuild_data.py      # about 1 min, creates data/raw (needs about 6 GB free)
jupyter notebook Signature_Validation.ipynb
```

The notebook's setup cell runs the rebuild for you, so the script step is optional. On **Colab**, run `!git clone …` and `%cd Signature_Validation_AI` first, then run the notebook top to bottom.

### If cloning is throttled or fails

The parts in `archives/` are plain files, so you can download them one at a time from GitHub (open a file, then **Download raw file**) and resume wherever you stopped. Put each set in its matching `archives/data/` or `archives/outputs/` folder, then run `python scripts/rebuild_data.py`. You can also open the `.001` file in 7-Zip by hand: you will get a zip, and inside it is `data/signature-rec-colab.zip`, which holds the images.

A shallow clone also downloads less:

```bash
git clone --depth 1 https://github.com/Hergli-Med-Amine/Signature_Validation_AI.git
```
