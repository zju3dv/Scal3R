from scal3r.utils.loop.models import aggregators, backbones

def get_backbone(backbone_arch: str = "dinov2_vitb14", backbone_config: dict | None = None):
    backbone_config = {} if backbone_config is None else dict(backbone_config)
    if "dinov2" not in backbone_arch.lower():
        raise ValueError(f"Unsupported loop backbone: {backbone_arch}")
    return backbones.DINOv2(model_name=backbone_arch, **backbone_config)


def get_aggregator(agg_arch: str = "SALAD", agg_config: dict | None = None):
    agg_config = {} if agg_config is None else dict(agg_config)
    if "salad" not in agg_arch.lower():
        raise ValueError(f"Unsupported loop aggregator: {agg_arch}")
    return aggregators.SALAD(**agg_config)
