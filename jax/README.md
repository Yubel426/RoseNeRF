# This is the code release for RoseNeRF based on [MultiNeRF](https://github.com/google-research/multinerf)

## Setup

```
# Clone the repo.
git clone https://github.com/Yubel426/RoseNeRF.git
cd rosenerf/jax/

# Make a conda environment.
conda create --name rosenerf python=3.9
conda activate rosenerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Running 

Example scripts for training, evaluating and rendering can be found in `scripts/`. And we evaluate the PSNR, SSIM, and LPIPS using our own script.

## Evaluating

We use PyTorch-based Python scripts to evaluate all our results, as there are many convient packages to use. Experiments on synthetic part of our dataset are evaluated using `../eval_metrics_syn.py`, and those on real captured part are using `../eval_metrics_llff.py`. You just need to set the variables `path` and `gt_path_root` in the scripts.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

### Notification

We make minimal modifications to this repository, therefore, all the other properties should remain the same. Please follow the [instructions](https://github.com/google-research/multinerf/blob/main/README.md) to further conduct researches.
