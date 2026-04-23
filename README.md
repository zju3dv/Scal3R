<div align="center">

<h1>
  <img src="assets/logo.png" alt="Scal3R logo" width="42" style="vertical-align: -0.18em; margin-right: 0.12em;">
  Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction
</h1>

### CVPR 2026 Highlight

<a href="https://github.com/xbillowy">Tao Xie</a><sup>1,2</sup>,
<a href="https://github.com/PeiPei233/">Peishan Yang</a><sup>1</sup>,
<a href="https://krahets.com/">Yudong Jin</a><sup>1</sup>,
Yingfeng Cai<sup>2</sup>,
<a href="https://yvanyin.xyz/">Wei Yin</a><sup>2</sup>,
Weiqiang Ren<sup>2</sup>,
Qian Zhang<sup>2</sup>,
<br>
Wei Hua<sup>3</sup>,
<a href="https://pengsida.net/">Sida Peng</a><sup>1</sup>,
<a href="https://github.com/xy-guo">Xiaoyang Guo</a><sup>2†</sup>,
<a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1†</sup>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2604.08542)
[![Safari](https://img.shields.io/badge/Project-Page-green?logo=safari&logoColor=fff)](https://zju3dv.github.io/scal3r)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Checkpoint-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/xbillowy/Scal3R)

</div>

***News***

- 2026-04-23: Release point cloud and camera pose visualization tools.
- 2026-04-17: Inference acceleration is enabled.
- 2026-04-10: The inference code is released.
- 2026-04-10: [Scal3R](https://zju3dv.github.io/scal3r/) has been selected as a highlight paper for CVPR 2026.

<p align="center">
  <img src="assets/teaser.gif" alt="Scal3R default teaser" width="100%">
</p>

## Installation

Use the automated installation script:

```bash
bash scripts/install.sh
```

The script creates or reuses a conda environment named `scal3r`, installs the core dependencies from `requirements.txt`, and installs Scal3R in editable mode. By default it uses `uv pip` inside that conda environment, with a plain `pip` fallback available.

This release currently includes inference only; evaluation and benchmark code are not part of the public package yet.

For detailed installation instructions and PyTorch/CUDA guidance, see [docs/install.md](docs/install.md).

Download the required checkpoints to `data/checkpoints/`:

```bash
mkdir -p data/checkpoints
hf download xbillowy/Scal3R scal3r.pt --repo-type model --local-dir data/checkpoints
curl -L https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt -o data/checkpoints/dino_salad.ckpt
```

## Usage

Run inference on a folder of images:

```bash
python -m scal3r.run --input_dir /path/to/images
```

You can also set an explicit tag or output directory:

```bash
python -m scal3r.run \
  --input_dir /path/to/images \
  --tag demo \
  --output_dir data/result/custom/demo
```

Important arguments:

- `--config`: model config path. Defaults to `configs/models/scal3r.yaml`.
- `--tag`: controls the default output directory name when `--output_dir` is not set.
- `--block_size` and `--overlap_size`: control chunking for long-sequence inference.
- `--save_dpt` and `--save_xyz`: control whether depth maps and point clouds are exported.
- `--offload_batches`, `--offload_outputs`: control whether to offload batches and outputs to disk.

By default, inference results are written to `data/result/custom/<tag>/`, and runtime artifacts are written to `data/result/custom/<tag>/runtime/`. The result directory typically contains:

- `mat.txt` for the predicted camera poses (camera-to-world transform matrix), each row is a raveled 4x4 matrix
- `intri.yml` and `extri.yml` for [EasyVolcap](https://github.com/zju3dv/EasyVolCap) format camera parameters
- `depths/` when depth export is enabled
- `points/` when point-cloud export is enabled
- `runtime/` for runtime artifacts

## TODOs

- [x] TODO: Release inference code.
- [ ] TODO: Release evaluation code along with dataset preparation scripts.
- [ ] TODO: Provide a simple viser viewer for the inference results.

## Acknowledgments

This project builds on and benefits from several excellent open-source works, especially [VGGT](https://github.com/facebookresearch/vggt), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), and [LaCT](https://github.com/a1600012888/LaCT). We thank the authors for making their code and ideas publicly available.

## Citation

```bibtex
@misc{xie2026scal3rscalabletesttimetraining,
      title={Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction}, 
      author={Tao Xie and Peishan Yang and Yudong Jin and Yingfeng Cai and Wei Yin and Weiqiang Ren and Qian Zhang and Wei Hua and Sida Peng and Xiaoyang Guo and Xiaowei Zhou},
      year={2026},
      eprint={2604.08542},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.08542}, 
}
```
