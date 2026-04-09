import torch
from torch import nn
from copy import deepcopy

from scal3r.engine import load_config
from scal3r.engine.path import resolve_release_path

from scal3r.utils.ttt_utils import TTTOperator
from scal3r.utils.vggt.models import Aggregator
from scal3r.utils.vggt.heads import CameraHead, DPTHead
from scal3r.utils.console_utils import get_logger, log_exceptions
from scal3r.utils.base_utils import DotDict as dotdict, to_plain_dict

logger = get_logger("scal3r.models.scal3r")
dataset_cfg_defaults = dotdict(
    render_ratio=1.0,
    rot90=None,
    proc_max_size=-1,
    proc_align_size=1,
    center_crop=True,
    focal_ratio=1.0,
    cam_param_type="abs_quat_fov",
    use_world_coord=True,
)


def get_nested(cfg, keys, default=None):
    current = cfg
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def default_ttt_order() -> list[TTTOperator]:
    return [
        TTTOperator(
            s=0,
            e=None,
            update=True,
            apply=False,
            use_cached=True,
            cache_last=True,
        ),
        TTTOperator(
            s=0,
            e=None,
            update=False,
            apply=True,
            use_cached=True,
            cache_last=True,
        ),
    ]


def load_checkpoint(model: nn.Module, ckpt_path: str):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint from %s", ckpt_path)


def resolve_checkpoint_path(cfg, checkpoint_path: str | None) -> str:
    ckpt_path = checkpoint_path or str(get_nested(cfg, ["model", "checkpoint"], "") or "").strip()
    if not ckpt_path:
        config_path = get_nested(cfg, ["__config_path__"], "<unknown config>")
        raise ValueError(
            "A sampler-only checkpoint is required for release inference. "
            f"Set `model.checkpoint` in {config_path} or pass `--checkpoint`."
        )
    return resolve_release_path(ckpt_path)


def extract_sampler_cfg(cfg) -> dotdict:
    sampler_cfg = dotdict(to_plain_dict(get_nested(cfg, ["model_cfg", "sampler_cfg"], {})))
    sampler_cfg.pop("_delete_", None)
    return sampler_cfg


def _select_cfg_keys(raw_cfg: dotdict, defaults: dotdict) -> dotdict:
    return dotdict({key: raw_cfg.get(key, value) for key, value in defaults.items()})


def _get_dataset_cfg_node(cfg) -> dotdict:
    raw_cfg = get_nested(cfg, ["val_dataloader_cfg", "dataset_cfg"], None)
    if raw_cfg is None:
        raw_cfg = get_nested(cfg, ["dataloader_cfg", "dataset_cfg"], {})
    return dotdict(to_plain_dict(raw_cfg))


class Scal3R(nn.Module):
    def __init__(self, sampler_cfg: dotdict):
        super().__init__()

        use_checkpoint = bool(sampler_cfg.get("use_checkpoint", False))
        use_reentrant = bool(sampler_cfg.get("use_reentrant", False))
        use_chunkwise_checkpoint = bool(sampler_cfg.get("use_chunkwise_checkpoint", False))
        dpt_head_use_checkpoint = bool(sampler_cfg.get("dpt_head_use_checkpoint", False))

        self.agg_regator_cfg = dotdict(
            deepcopy(
                sampler_cfg.get(
                    "agg_regator_cfg",
                    {
                        "img_size": 518,
                        "patch_size": 14,
                        "embed_dim": 1024,
                    },
                )
            )
        )
        self.cam_decoder_cfg = dotdict(
            deepcopy(sampler_cfg.get("cam_decoder_cfg", {"dim_in": 2048}))
        )
        self.xyz_decoder_cfg = dotdict(
            deepcopy(
                sampler_cfg.get(
                    "xyz_decoder_cfg",
                    {
                        "dim_in": 2048,
                        "output_dim": 4,
                        "activation": "inv_log",
                        "conf_activation": "expp1",
                    },
                )
            )
        )
        self.dpt_decoder_cfg = dotdict(
            deepcopy(
                sampler_cfg.get(
                    "dpt_decoder_cfg",
                    {
                        "dim_in": 2048,
                        "output_dim": 2,
                        "activation": "exp",
                        "conf_activation": "expp1",
                    },
                )
            )
        )

        if "embed_dim" in self.agg_regator_cfg:
            embed_dim = int(self.agg_regator_cfg["embed_dim"])
            self.cam_decoder_cfg["dim_in"] = embed_dim * 2
            self.xyz_decoder_cfg["dim_in"] = embed_dim * 2
            self.dpt_decoder_cfg["dim_in"] = embed_dim * 2

        if "intermediate_layer_idx" in self.agg_regator_cfg:
            intermediate_layer_idx = deepcopy(self.agg_regator_cfg["intermediate_layer_idx"])
            self.xyz_decoder_cfg["intermediate_layer_idx"] = intermediate_layer_idx
            self.dpt_decoder_cfg["intermediate_layer_idx"] = intermediate_layer_idx

        self.agg_regator = Aggregator(
            **self.agg_regator_cfg,
            use_checkpoint=use_checkpoint,
            use_reentrant=use_reentrant,
            use_chunkwise_checkpoint=use_chunkwise_checkpoint,
        )
        self.cam_decoder = CameraHead(**self.cam_decoder_cfg)
        self.xyz_decoder = DPTHead(**self.xyz_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint)
        self.dpt_decoder = DPTHead(**self.dpt_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint)
        self.ttt_order = default_ttt_order()


def extract_dataset_cfg(cfg) -> dotdict:
    return _select_cfg_keys(_get_dataset_cfg_node(cfg), dataset_cfg_defaults)


@log_exceptions(logger, "Unhandled exception while building SCAL3R model")
def build_sampler_from_config(
    config_path: str,
    device: torch.device,
    checkpoint_path: str | None = None,
):
    cfg = load_config(config_path)
    sampler_cfg = extract_sampler_cfg(cfg)
    sampler = Scal3R(sampler_cfg)
    load_checkpoint(sampler, resolve_checkpoint_path(cfg, checkpoint_path))
    sampler = sampler.to(device)
    sampler.eval()

    logger.info("Built release sampler: %s", sampler.__class__.__name__)
    return sampler, extract_dataset_cfg(cfg)
