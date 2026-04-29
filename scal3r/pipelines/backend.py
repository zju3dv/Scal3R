import gc
import os
import torch
import argparse
import numpy as np
from typing import List
from os.path import join
import time as wall_time
from copy import deepcopy
from einops import rearrange

from scal3r.utils.ray_utils import get_rays
from scal3r.utils.data_utils import to_cuda
from scal3r.utils.result_utils import save_results
from scal3r.models import build_sampler_from_config
from scal3r.utils.base_utils import DotDict as dotdict
from scal3r.utils.cam_utils import decode_camera_params
from scal3r.utils.math_utils import affine_inverse, affine_padding
from scal3r.engine.path import get_default_output_dir, resolve_release_path
from scal3r.utils.console_utils import get_logger, log_block, log_exceptions, tqdm
from scal3r.utils.runtime_utils import StageRecorder, StopAfterStage, maybe_stop_after, release_memory
from scal3r.utils.image_utils import build_image_only_block, collect_image_paths, load_and_preprocess_images
from scal3r.utils.loop import accumulate_transform, build_loop_batches, build_map_processor, build_sim3_loop_optimizer, combine_transform, detect_loops, visualize_loop
from scal3r.utils.offload_utils import cleanup_offload_root, clear_dpt_state, get_offload_root, get_runtime_root, materialize_payload, materialize_tensor_dict, offload_batch_block, offload_output_block, persist_dpt_state, remove_payload, should_release_runtime_state, store_agg_state


logger = get_logger("scal3r.backend")


def collect_intermediate_layers(model, start_index: int) -> list[int]:
    return [
        start_index + offset
        for offset in range(model.agg_regator.aa_block_size)
        if start_index + offset in model.agg_regator.intermediate_layer_idx
    ]


def format_runtime_config(args) -> list[str]:
    return [
        f"block_size: {args.block_size}",
        f"overlap_size: {args.overlap_size}",
        f"use_loop: {bool(args.use_loop)}",
        f"preprocess_workers: {int(args.preprocess_workers)}",
        f"max_align_points_per_frame: {args.max_align_points_per_frame}",
        f"pgo_workers: {int(args.pgo_workers)}",
        f"save_dpt: {bool(args.save_dpt)}",
        f"save_xyz: {bool(args.save_xyz)}",
        f"streaming_state: {bool(args.streaming_state)}",
        f"result_dir: {args.result_dir}",
        f"runtime_dir: {args.runtime_dir}",
        f"offload_batches: {bool(args.offload_batches)}",
        f"offload_outputs: {bool(args.offload_outputs)}",
        f"probe_dir: {args.probe_dir or ''}",
        f"stop_after_stage: {args.stop_after_stage or ''}",
    ]


def load_data(dataset_cfg: dotdict, args, recorder: StageRecorder | None = None):
    if recorder is not None:
        recorder.record('collect_images.begin', input_dir=args.input_dir, image_patterns=args.image_patterns)
    image_paths = collect_image_paths(args.input_dir, args.image_patterns)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if recorder is not None:
        recorder.record(
            'collect_images.done',
            image_count=len(image_paths),
            first_image=image_paths[0] if image_paths else '',
            last_image=image_paths[-1] if image_paths else '',
        )
        maybe_stop_after('collect_images.done', args, recorder)

        recorder.record(
            'preprocess_images.begin',
            image_count=len(image_paths),
            preprocess_workers=int(args.preprocess_workers),
        )
    sequence, height, width = load_and_preprocess_images(
        image_paths,
        dataset_cfg,
        preprocess_workers=args.preprocess_workers,
    )
    if recorder is not None:
        recorder.record(
            'preprocess_images.done',
            image_count=len(sequence),
            height=int(height),
            width=int(width),
            preprocess_workers=int(args.preprocess_workers),
        )
        maybe_stop_after('preprocess_images.done', args, recorder)

    n_samples = len(sequence)
    block_size, overlap_size = args.block_size, args.overlap_size
    assert block_size > overlap_size, f'[ERROR] block_size {block_size} must be larger than overlap_size {overlap_size}'

    n_srcs = block_size
    n_blocks = (n_samples - overlap_size + (n_srcs - overlap_size) - 1) // (n_srcs - overlap_size)
    if n_blocks == 0 or n_samples <= block_size:
        block_size = n_samples
        n_srcs = n_samples
        n_blocks = 1

    batches, indices = [], []
    batch_index = 0

    def build_block(sequence, block_indices: List[int], block_height: int, block_width: int, block_overlap_size: int, block_dataset_cfg: dotdict):
        nonlocal batch_index
        batch = build_image_only_block(sequence, block_indices, block_height, block_width, block_overlap_size, block_dataset_cfg)
        if args.offload_batches:
            batch = offload_batch_block(batch, args, batch_index)
            release_memory(args.device)
        batch_index += 1
        return batch

    if recorder is not None:
        recorder.record(
            'build_blocks.begin',
            n_samples=int(n_samples),
            block_size=int(block_size),
            overlap_size=int(overlap_size),
            n_blocks=int(n_blocks),
        )
    pbar = tqdm(total=n_blocks, desc='Loading image blocks')
    for i in range(n_blocks):
        sampler_index = i * (n_srcs - overlap_size)
        block_indices = list(range(sampler_index, min(sampler_index + n_srcs, n_samples)))
        batches.append(build_block(sequence, block_indices, height, width, overlap_size, dataset_cfg))
        indices.append((sampler_index, min(sampler_index + n_srcs, n_samples)))
        pbar.update()
    pbar.close()
    if recorder is not None:
        recorder.record(
            'build_blocks.done',
            n_blocks=int(len(batches)),
            offload_batches=bool(args.offload_batches),
            offload_root=get_offload_root(args) if args.offload_batches else '',
        )
        maybe_stop_after('build_blocks.done', args, recorder)

    args.n_blocks = n_blocks
    logger.info('Loading finished, %d blocks are loaded from %d images', len(batches), len(image_paths))

    if args.use_loop:
        if recorder is not None:
            recorder.record('loop_detection.begin', image_count=len(image_paths))
        os.makedirs(get_runtime_root(args), exist_ok=True)
        loop_list = detect_loops(
            image_paths,
            result_dir=get_runtime_root(args),
            loop_ckpt=args.loop_ckpt,
            nms_threshold=25,
            min_frame_gap=10,
        )
        batches_loop, indices_loop = build_loop_batches(
            sequence,
            indices,
            loop_list,
            args.loop_size // 2,
            height,
            width,
            dataset_cfg,
            build_block,
            logger.info,
        )
        batches.extend(batches_loop)
        indices.extend(indices_loop)
        args.n_blocks_loop = len(batches_loop)
        if recorder is not None:
            recorder.record(
                'loop_detection.done',
                loop_pairs=int(len(loop_list)),
                loop_blocks=int(len(batches_loop)),
            )
            maybe_stop_after('loop_detection.done', args, recorder)

    del sequence
    release_memory(args.device)
    return batches, indices


def apply_ttt(model, agg_state_refs, dpt_state_refs, args, layer_index: int, dpt_layer_set: set[int]):
    ttt_order_grad = deepcopy(model.ttt_order[0:1])
    ttt_order_grad = [order._replace(use_cached=False)._replace(cache_last=False) for order in ttt_order_grad]

    ttt_order_apply = deepcopy(model.ttt_order[-1:])
    ttt_order_apply = [order._replace(use_cached=False)._replace(cache_last=False) for order in ttt_order_apply]

    w0_grad_sum, w1_grad_sum, w2_grad_sum = None, None, None
    shared_tokens = None

    for block_index in range(len(agg_state_refs)):
        agg_state = materialize_payload(agg_state_refs[block_index])
        if shared_tokens is None:
            shared_tokens = agg_state['tokens']
        g0, g1, g2 = model.agg_regator.ttt_gradient(
            index=layer_index, ttt_order=ttt_order_grad, **to_cuda(agg_state, args.device),
        )
        if w0_grad_sum is None:
            w0_grad_sum, w1_grad_sum, w2_grad_sum = g0, g1, g2
        else:
            w0_grad_sum.add_(g0)
            w1_grad_sum.add_(g1)
            w2_grad_sum.add_(g2)
        del agg_state
        del g0, g1, g2
        if should_release_runtime_state(args):
            release_memory(args.device)

    shared_w0, shared_w1, shared_w2 = model.agg_regator.ttt_update(
        index=layer_index,
        tokens=shared_tokens,
        w0_grad=w0_grad_sum,
        w1_grad=w1_grad_sum,
        w2_grad=w2_grad_sum,
        ttt_order=ttt_order_grad,
    )
    del shared_tokens

    for block_index in range(len(agg_state_refs)):
        agg_state = materialize_payload(agg_state_refs[block_index])
        temp_output = (
            {layer_index: materialize_payload(dpt_state_refs[block_index][layer_index])}
            if layer_index in dpt_layer_set
            else None
        )
        updated_state = model.agg_regator.ttt_apply(
            index=layer_index,
            ttt_order=ttt_order_apply,
            output=temp_output,
            w0=shared_w0,
            w1=shared_w1,
            w2=shared_w2,
            **to_cuda(agg_state, args.device),
        )
        agg_state_refs[block_index] = store_agg_state(updated_state, args, block_index)
        if temp_output is not None:
            persist_dpt_state(temp_output, dpt_state_refs[block_index], args, block_index)
        del agg_state, updated_state, temp_output
        if should_release_runtime_state(args):
            release_memory(args.device)

    del w0_grad_sum, w1_grad_sum, w2_grad_sum
    del shared_w0, shared_w1, shared_w2
    if should_release_runtime_state(args) and args.device.startswith('cuda'):
        torch.cuda.empty_cache()


def forward(model, batches, args, recorder: StageRecorder | None = None):
    output = [None for _ in range(len(batches))]

    batch0 = materialize_payload(batches[0])
    B, S = batch0.meta.rgb.shape[:2]
    H, W = batch0.meta.H[0].item(), batch0.meta.W[0].item()
    N = len(batches)
    del batch0
    assert B == 1, f'[ERROR] this implementation only supports B=1 for sequential inference, got B={B}.'

    agg_state_refs = [None for _ in range(len(batches))]
    dpt_state_refs = [dict() for _ in range(len(batches))]
    dpt_layer_set = set(model.agg_regator.intermediate_layer_idx)

    if recorder is not None:
        recorder.record('embedder.begin', n_blocks=int(N), frames_per_block=int(S), height=int(H), width=int(W))
    pbar = tqdm(total=N, desc='Forward DINOv2 embedder')
    for b, batch_ref in enumerate(batches):
        batch = materialize_payload(batch_ref)
        rgb = rearrange(to_cuda(batch.meta.rgb, args.device), 'b n (h w) c -> b n c h w', h=H, w=W)
        agg_state = model.agg_regator.prepare(rgb)
        agg_state_refs[b] = store_agg_state(agg_state, args, b)
        block_tokens = int(agg_state['tokens'].shape[1]) if 'tokens' in agg_state else int(S)
        del batch, rgb, agg_state
        if should_release_runtime_state(args):
            release_memory(args.device)
        if recorder is not None:
            recorder.record(
                f'embedder_block_{b:02d}.done',
                block_index=int(b),
                block_frames=block_tokens,
            )
        pbar.update()
    pbar.close()
    if recorder is not None:
        recorder.record('embedder.done', n_blocks=int(N))
        maybe_stop_after('embedder.done', args, recorder)

    if recorder is not None:
        recorder.record(
            'aggregator.begin',
            n_layers=int(model.agg_regator.aa_block_num),
            ttt_layers=[int(i) for i in model.agg_regator.ttt_layer_idx],
        )
    pbar = tqdm(total=model.agg_regator.aa_block_num, desc='Forward aggregator')
    for j in range(model.agg_regator.aa_block_num):
        forward_intermediate_layers = collect_intermediate_layers(model, j)
        need_forward_outputs = len(forward_intermediate_layers) > 0
        if recorder is not None:
            recorder.record(f'aggregator_layer_{j:02d}.begin', layer_index=int(j))
        for b, _ in enumerate(batches):
            agg_state = materialize_payload(agg_state_refs[b])
            temp_output = {} if need_forward_outputs else None
            updated_state = model.agg_regator.forward_layer(
                index=j, output=temp_output, **to_cuda(agg_state, args.device),
            )
            agg_state_refs[b] = store_agg_state(updated_state, args, b)
            if temp_output is not None:
                persist_dpt_state(temp_output, dpt_state_refs[b], args, b)
            del agg_state, updated_state, temp_output
            if should_release_runtime_state(args):
                release_memory(args.device)

        with torch.amp.autocast('cuda', enabled=False):
            if (model.agg_regator.frame_use_ttt or model.agg_regator.global_use_ttt) and j in model.agg_regator.ttt_layer_idx:
                apply_ttt(model, agg_state_refs, dpt_state_refs, args, j, dpt_layer_set)

        pbar.update()
        if recorder is not None:
            recorder.record(
                f'aggregator_layer_{j:02d}.done',
                layer_index=int(j),
                uses_ttt=bool((model.agg_regator.frame_use_ttt or model.agg_regator.global_use_ttt) and j in model.agg_regator.ttt_layer_idx),
            )
            maybe_stop_after(f'aggregator_layer_{j:02d}.done', args, recorder)
    pbar.close()
    if recorder is not None:
        recorder.record('aggregator.done', n_layers=int(model.agg_regator.aa_block_num))
        maybe_stop_after('aggregator.done', args, recorder)

    for b, dpt_state in enumerate(dpt_state_refs):
        dpt_state[-1] = dpt_state[model.agg_regator.depth - 1]
        remove_payload(agg_state_refs[b])
        agg_state_refs[b] = None

    if recorder is not None:
        recorder.record('decoder.begin', n_blocks=int(N))
    with torch.amp.autocast('cuda', enabled=False):
        pbar = tqdm(total=N, desc='Forward decoder')
        for b, batch_ref in enumerate(batches):
            if recorder is not None:
                recorder.record(f'decoder_block_{b:02d}.begin', block_index=int(b))
            batch = materialize_payload(batch_ref)
            rgb_feats = to_cuda(materialize_tensor_dict(dpt_state_refs[b]), args.device)
            rgb = rearrange(batch.meta.rgb, 'b n (h w) c -> b n c h w', h=H, w=W).to(args.device)

            cam_maps = model.cam_decoder(rgb_feats)
            xyz_map, xyz_cnf = model.xyz_decoder(
                rgb_feats,
                images=rgb,
                patch_start_idx=model.agg_regator.patch_start_idx,
            )
            dpt_map, dpt_cnf = model.dpt_decoder(
                rgb_feats,
                images=rgb,
                patch_start_idx=model.agg_regator.patch_start_idx,
            )

            xyz_map = xyz_map[..., :H, :W, :]
            dpt_map = dpt_map[..., :H, :W, :]
            xyz_cnf = xyz_cnf[..., :H, :W]
            dpt_cnf = dpt_cnf[..., :H, :W]

            cam_map = cam_maps[-1].cpu()
            block_output = dotdict(
                cam_map=cam_map[..., :9],
                xyz_map=rearrange(xyz_map, 'b s h w c -> b s (h w) c').cpu(),
                dpt_map=rearrange(dpt_map, 'b s h w c -> b s (h w) c').cpu(),
                xyz_cnf=rearrange(xyz_cnf, 'b s h w -> b s (h w) 1').cpu(),
                dpt_cnf=rearrange(dpt_cnf, 'b s h w -> b s (h w) 1').cpu(),
            )
            if cam_map.shape[-1] > 9:
                block_output.scale = cam_map[..., -1].mean(dim=-1)
            if args.offload_outputs:
                output[b] = offload_output_block(block_output, args, b)
            else:
                output[b] = block_output

            clear_dpt_state(dpt_state_refs[b])
            del batch, block_output, cam_map
            del cam_maps, xyz_map, xyz_cnf, dpt_map, dpt_cnf, rgb_feats, rgb
            if should_release_runtime_state(args):
                release_memory(args.device)
            if recorder is not None:
                recorder.record(
                    f'decoder_block_{b:02d}.done',
                    block_index=int(b),
                    offload_outputs=bool(args.offload_outputs),
                )
                maybe_stop_after(f'decoder_block_{b:02d}.done', args, recorder)

            pbar.update()
        pbar.close()
    if recorder is not None:
        recorder.record('decoder.done', n_blocks=int(N))
        maybe_stop_after('decoder.done', args, recorder)

    return output


def post_process(
    raw: dotdict,
    batches: list,
    indices: list,
    args,
    n_blocks_loop: int = 0,
    alignment: str = 'sim3_wet',
    use_xyz_align: int = 1,
    recorder: StageRecorder | None = None,
):
    i_src = []
    processed = dotdict(output=dotdict(c2w=[], ixt=[]))
    if args.save_dpt:
        processed.output.dpt_map = []
    visualize = dotdict(
        block_xyz=[], block_rgb=[], block_msk=[],
        world_xyz=[], world_rgb=[], world_msk=[],
    )
    batch0 = materialize_payload(batches[0])
    height, width = batch0.meta.H[0].item(), batch0.meta.W[0].item()
    del batch0

    logger.info('Using %s for pose graph optimization', alignment)
    map_processor = build_map_processor(
        alignment,
        max_align_points_per_frame=args.max_align_points_per_frame,
    )
    if recorder is not None:
        recorder.record(
            'post_process.begin',
            n_blocks=int(len(batches)),
            n_blocks_loop=int(n_blocks_loop),
            alignment=alignment,
        )

    def prepare(batch, output):
        dpt = output.dpt_map[0].cpu()
        cnf = output.dpt_cnf[0].cpu()
        if use_xyz_align == 0:
            w2c, ixt = decode_camera_params(
                output.cam_map[0], height, width,
                batch.meta.cam_param_type[0],
                not batch.meta.use_world_coord[0].item(),
            )
            ray_o, ray_d = get_rays(
                height, width, ixt.cpu(), w2c[..., :3, :3].cpu(), w2c[..., :3, 3:].cpu(),
                z_depth=True, correct_pix=True,
            )
            ray_o = ray_o.reshape(w2c.shape[0], -1, 3)
            ray_d = ray_d.reshape(w2c.shape[0], -1, 3)
            xyz = ray_o + ray_d * dpt
        else:
            xyz = output.xyz_map[0].cpu()
            cnf = output.xyz_cnf[0].cpu()
        if 'scale' in output:
            scale = output.scale[0][None, None, None].cpu()
            xyz = xyz * scale
            dpt = dpt * scale
        return xyz.numpy(), dpt.numpy(), cnf.numpy()

    norm_track, loop_track = [], []
    n_blocks = len(batches) if n_blocks_loop == 0 else len(batches) - n_blocks_loop

    # Phase 1: Create all submaps (mask computation + point filtering)
    pbar = tqdm(total=n_blocks, desc='Preparing submaps')
    for b in range(n_blocks):
        batch = materialize_payload(batches[b])
        raw_block = materialize_payload(raw[b])
        xyz, dpt, cnf = prepare(batch, raw_block)
        map_processor.add_submap(
            xyz=xyz,
            dpt=dpt,
            cnf=cnf,
            msk=batch.msk[0].cpu().numpy(),
            file_name=batch.src_inds[0].cpu().numpy().tolist(),
            compute_constraint=False,
        )
        del batch, raw_block, xyz, dpt, cnf
        if args.offload_batches or args.offload_outputs:
            release_memory(args.device)
        pbar.update()
    pbar.close()

    # Phase 2: Parallel pairwise alignment (NumPy BLAS releases GIL)
    n_pairs = max(0, n_blocks - 1)
    if args.pgo_workers > 0:
        n_workers = min(n_pairs, args.pgo_workers)
    else:
        n_workers = min(n_pairs, os.cpu_count() or 4)
    logger.info('Aligning %d adjacent block pairs in parallel (%d workers)', n_pairs, n_workers)
    norm_track = map_processor.align_submaps_parallel(max_workers=n_workers)

    if n_blocks_loop > 0:
        pbar = tqdm(total=n_blocks_loop, desc='Processing loop closure blocks')
        for k in range(n_blocks_loop):
            block1 = len(batches) - n_blocks_loop + k
            block0, _, block2, _ = indices[block1]

            processor0 = build_map_processor(
                alignment,
                max_align_points_per_frame=args.max_align_points_per_frame,
            )
            processor2 = build_map_processor(
                alignment,
                max_align_points_per_frame=args.max_align_points_per_frame,
            )
            batch0 = materialize_payload(batches[block0])
            batch1 = materialize_payload(batches[block1])
            batch2 = materialize_payload(batches[block2])
            raw0 = materialize_payload(raw[block0])
            raw1 = materialize_payload(raw[block1])
            raw2 = materialize_payload(raw[block2])
            xyz0, dpt0, cnf0 = prepare(batch0, raw0)
            xyz1, dpt1, cnf1 = prepare(batch1, raw1)
            xyz2, dpt2, cnf2 = prepare(batch2, raw2)

            processor0.add_submap(
                xyz=xyz0, dpt=dpt0, cnf=cnf0,
                msk=batch0.msk[0].cpu().numpy(),
                file_name=batch0.src_inds[0].cpu().numpy().tolist(),
            )
            s0, R0, t0 = processor0.add_submap(
                xyz=xyz1, dpt=dpt1, cnf=cnf1,
                msk=batch1.msk[0].cpu().numpy(),
                file_name=batch1.src_inds[0].cpu().numpy().tolist(),
            )
            processor2.add_submap(
                xyz=xyz2, dpt=dpt2, cnf=cnf2,
                msk=batch2.msk[0].cpu().numpy(),
                file_name=batch2.src_inds[0].cpu().numpy().tolist(),
            )
            s1, R1, t1 = processor2.add_submap(
                xyz=xyz1, dpt=dpt1, cnf=cnf1,
                msk=batch1.msk[0].cpu().numpy(),
                file_name=batch1.src_inds[0].cpu().numpy().tolist(),
            )

            s01, R01, t01 = combine_transform(s0, R0, t0, s1, R1, t1)
            loop_track.append((block0, block2, (s01, R01, t01)))
            del batch0, batch1, batch2
            del raw0, raw1, raw2
            del xyz0, xyz1, xyz2, dpt0, dpt1, dpt2, cnf0, cnf1, cnf2
            del processor0, processor2
            if args.offload_batches or args.offload_outputs:
                release_memory(args.device)
            pbar.update()
        pbar.close()

        optimizer = build_sim3_loop_optimizer()
        ini_track = optimizer.sequential_to_absolute_poses(norm_track)
        opt_track = optimizer.optimize(norm_track, loop_track)
        res_track = accumulate_transform(opt_track)
        vis_track = optimizer.sequential_to_absolute_poses(opt_track)
        visualize_loop(ini_track, vis_track, loop_track, get_runtime_root(args))

        raw = raw[:n_blocks]
        batches = batches[:n_blocks]
        indices = indices[:n_blocks]
    else:
        res_track = [map_processor.optimizer.get_submap(i).global_pose for i in range(len(batches))]

    del map_processor
    gc.collect()

    overlap_prev_inds, overlap_curr_inds = [], []
    for b, batch_ref in enumerate(batches):
        if b == len(batches) - 1:
            continue
        batch = materialize_payload(batch_ref)
        next_batch = materialize_payload(batches[b + 1])
        prev_src_inds = batch.src_inds[0].tolist()
        curr_src_inds = next_batch.src_inds[0].tolist()
        overlap_prev_inds.append(torch.as_tensor([
            prev_src_inds.index(idx) for idx in batch.orig_src_inds[0].tolist() if idx in curr_src_inds
        ], dtype=torch.long).numpy().tolist())
        overlap_curr_inds.append(torch.as_tensor([
            curr_src_inds.index(idx) for idx in batch.orig_src_inds[0].tolist() if idx in curr_src_inds
        ], dtype=torch.long).numpy().tolist())
        del batch, next_batch

    pbar = tqdm(total=len(batches), desc='Getting aligned results')
    for b, batch_ref in enumerate(batches):
        batch = materialize_payload(batch_ref)
        raw_block = materialize_payload(raw[b])
        S = raw_block.cam_map[0].shape[0]
        j = b
        overlap_size = batch.meta.overlap_size[0].item()
        w2c, ixt = decode_camera_params(
            raw_block.cam_map[0], height, width,
            batch.meta.cam_param_type[0],
            not batch.meta.use_world_coord[0].item(),
        )

        if 'scale' in raw_block:
            scale = raw_block.scale[0][None, None, None].cpu()
            w2c[..., :3, 3:] = w2c[..., :3, 3:] * scale.to(w2c.device)

        inds = torch.arange(S).numpy().tolist()
        if j % 2 == 0 and j < len(batches) - 1:
            midx = [k for k in inds if k not in overlap_prev_inds[j][overlap_size // 2:]]
            if j > 0:
                midx = [k for k in midx if k not in overlap_curr_inds[j - 1][:overlap_size // 2]]
        elif j % 2 == 1 and j > 0:
            midx = [k for k in inds if k not in overlap_curr_inds[j - 1][:overlap_size // 2]]
            if j < len(batches) - 1:
                midx = [k for k in midx if k not in overlap_prev_inds[j][overlap_size // 2:]]
        elif j % 2 == 0 and j == len(batches) - 1 and len(batches) > 1:
            midx = [k for k in inds if k not in overlap_curr_inds[j - 1][:overlap_size // 2]]
        else:
            midx = inds
        i_src.extend(batch.src_inds[0][midx].cpu().numpy().tolist())

        processed.output.c2w.append(
            res_track[j][None] @ affine_inverse(affine_padding(w2c[midx])).cpu().numpy()
        )
        processed.output.ixt.append(ixt[midx].cpu().numpy())
        if args.save_dpt:
            processed.output.dpt_map.append(raw_block.dpt_map[0][midx].cpu().numpy())

        if args.save_xyz:
            block_xyz, block_rgb, block_msk = [], [], []
            xyz, _, cnf = prepare(batch, raw_block)
            for k in range(raw_block.xyz_map[0].shape[0]):
                xyz_map = xyz[k]
                xyz_map = np.concatenate([xyz_map, np.ones_like(xyz_map[..., :1])], axis=-1)
                xyz_map = xyz_map @ res_track[j][:3].T

                msk = cnf[k][..., 0] > np.mean(cnf[k][..., 0]) * args.confidence_xyz_threshold
                xyz_map = xyz_map[msk]
                rgb_map = batch.meta.rgb[0][k].cpu().numpy()[msk]

                if k in midx:
                    visualize.world_xyz.append(xyz_map)
                    visualize.world_rgb.append(rgb_map)
                    visualize.world_msk.append(msk.reshape(height, width))

                block_xyz.append(xyz_map)
                block_rgb.append(rgb_map)
                block_msk.append(msk.reshape(height, width))

            visualize.block_xyz.append(block_xyz)
            visualize.block_rgb.append(block_rgb)
            visualize.block_msk.append(block_msk)

        del batch, raw_block
        if args.offload_batches or args.offload_outputs:
            release_memory(args.device)
        pbar.update()
    pbar.close()

    order = np.argsort(i_src, kind='stable')
    for key in processed.output.keys():
        processed.output[key] = np.concatenate(processed.output[key], axis=0)
        processed.output[key] = processed.output[key][order]
    if recorder is not None:
        recorder.record(
            'post_process.done',
            n_frames=int(processed.output.c2w.shape[0]),
            save_dpt=bool(args.save_dpt),
            save_xyz=bool(args.save_xyz),
        )
        maybe_stop_after('post_process.done', args, recorder)

    return processed, raw, batches, indices, visualize


def parse_args():
    parser = argparse.ArgumentParser(description='SCAL3R standalone inference backend')
    parser.add_argument('--config', type=str, default='configs/models/scal3r.yaml', help='Config file containing model_cfg.sampler_cfg and dataset preprocessing settings')
    parser.add_argument('--checkpoint', type=str, default='', help='Optional override for the sampler-only checkpoint (.pt)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input image sequence')
    parser.add_argument('--result_dir', type=str, default='', help='Directory to save final inference results. Defaults to data/result/custom/run.')
    parser.add_argument('--runtime_dir', type=str, default='', help='Directory to save runtime metadata and temporary artifacts. Defaults to <result_dir>/runtime.')
    parser.add_argument('--image_patterns', type=str, default='*.png,*.jpg,*.jpeg,*.bmp', help='Comma-separated glob patterns for input images')
    parser.add_argument('--max_images', type=int, default=-1, help='If positive, only use the first N sorted images from input_dir')
    parser.add_argument('--preprocess_workers', type=int, default=32, help='Number of worker threads used for image loading and preprocessing')
    parser.add_argument('--block_size', type=int, default=60, help='Number of frames in a block')
    parser.add_argument('--overlap_size', type=int, default=30, help='Number of overlapping frames between adjacent blocks')
    parser.add_argument('--loop_size', type=int, default=20, help='Window size for loop-closure block construction')
    parser.add_argument('--loop_ckpt', type=str, default='data/checkpoints/dino_salad.ckpt', help='Optional SALAD loop-detector checkpoint path; fallback loop detection is used when omitted')
    parser.add_argument('--use_xyz_align', type=int, default=0, help='Whether to align chunks using predicted xyz')
    parser.add_argument('--max_align_points_per_frame', type=int, default=None, help='Optional per-overlap-frame point cap for PGO alignment. None keeps all points.')
    parser.add_argument('--pgo_workers', type=int, default=32, help='Number of workers for adjacent-block PGO alignment. Use 0 for auto.')
    parser.add_argument('--test_use_amp', action='store_true', help='Whether to use AMP during inference')
    parser.add_argument('--save_dpt', type=int, default=1, help='Whether to save predicted depth maps')
    parser.add_argument('--save_xyz', type=int, default=1, help='Whether to save predicted point clouds')
    parser.add_argument('--downsample_xyz_ratio', type=float, default=0.15, help='Downsample ratio for saved point clouds')
    parser.add_argument('--confidence_xyz_threshold', type=float, default=0.75, help='Confidence threshold multiplier for saved point clouds')
    parser.add_argument('--use_loop', type=int, default=1, help='Whether to enable loop detection and loop blocks')
    parser.add_argument('--streaming_state', type=int, default=0, help='Whether to stream aggregator and decoder states through disk instead of keeping them in memory')
    parser.add_argument('--offload_batches', type=int, default=0, help='Whether to offload prepared blocks to disk between stages')
    parser.add_argument('--offload_outputs', type=int, default=0, help='Whether to offload decoded block outputs to disk')
    parser.add_argument('--cleanup_offload', type=int, default=1, help='Whether to remove temporary offload files after finishing')
    parser.add_argument('--offload_dir', type=str, default='', help='Optional directory for temporary offload files; defaults under runtime_dir/offload')
    parser.add_argument('--probe_dir', type=str, default='', help='Optional directory for step-by-step probe JSON outputs')
    parser.add_argument('--stop_after_stage', type=str, default='', help='Optional exact stage name after which the backend exits early')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device string for inference')
    return dotdict(vars(parser.parse_args()))


@log_exceptions(logger, "Unhandled exception in backend")
def main():
    args = parse_args()
    args.config = str(resolve_release_path(args.config))
    args.result_dir = str(resolve_release_path(args.result_dir or get_default_output_dir("run")))
    args.runtime_dir = str(resolve_release_path(args.runtime_dir) if args.runtime_dir else join(args.result_dir, 'runtime'))
    if args.checkpoint:
        args.checkpoint = str(resolve_release_path(args.checkpoint))
    if args.loop_ckpt:
        args.loop_ckpt = str(resolve_release_path(args.loop_ckpt))
    if args.offload_dir:
        args.offload_dir = str(resolve_release_path(args.offload_dir))
    if args.probe_dir:
        args.probe_dir = str(resolve_release_path(args.probe_dir))
    device = torch.device(args.device)
    recorder = StageRecorder(args.probe_dir or '', args.device)
    log_block("Runtime config: (", format_runtime_config(args), closing=")")

    try:
        if recorder.enabled:
            recorder.record(
                'process.begin',
                input_dir=args.input_dir,
                result_dir=args.result_dir,
                runtime_dir=args.runtime_dir,
                checkpoint=args.checkpoint,
                device=str(device),
            )
        if recorder is not None:
            recorder.record('load_model.begin', config=args.config, checkpoint=args.checkpoint or '')
        sampler, dataset_cfg = build_sampler_from_config(args.config, device, args.checkpoint)
        if recorder is not None:
            recorder.record(
                'load_model.done',
                checkpoint=args.checkpoint or '',
                model_name=getattr(sampler, '__class__', type(sampler)).__name__,
            )
            maybe_stop_after('load_model.done', args, recorder)

        batches, indices = load_data(dataset_cfg, args, recorder=recorder)
        if recorder is not None:
            recorder.record(
                'load_data.done',
                n_blocks=int(len(batches)),
                n_blocks_loop=int(args.get('n_blocks_loop', 0)),
            )
            maybe_stop_after('load_data.done', args, recorder)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        t0 = wall_time.time()

        amp_enabled = bool(args.test_use_amp and device.type == 'cuda')
        amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                output = forward(sampler, batches, args, recorder=recorder)

        processed, output, batches, indices, visualize = post_process(
            output,
            batches,
            indices,
            args,
            n_blocks_loop=args.get('n_blocks_loop', 0),
            alignment='sim3_wet',
            use_xyz_align=args.use_xyz_align,
            recorder=recorder,
        )

        t1 = wall_time.time()
        runtime = {
            'time': t1 - t0,
            'memory': torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == 'cuda' else 0.0,
            'n_frames': int(processed.output.c2w.shape[0]),
            'offload': {
                'states': bool(args.streaming_state),
                'batches': bool(args.offload_batches),
                'outputs': bool(args.offload_outputs),
                'cleanup': bool(args.cleanup_offload),
                'offload_dir': get_offload_root(args) if (bool(args.streaming_state) or bool(args.offload_batches) or bool(args.offload_outputs)) else '',
            },
            'runtime_dir': get_runtime_root(args),
            'runtime_json': os.path.join(get_runtime_root(args), 'runtime.json'),
        }
        runtime['fps'] = runtime['n_frames'] / max(runtime['time'], 1e-8)
        logger.info(
            'Inference finished, time cost: %.2fs, memory usage: %.2fGB, frames: %d',
            runtime["time"],
            runtime["memory"],
            runtime["n_frames"],
        )

        save_results(processed, batches, visualize, runtime, args, recorder=recorder)
        logger.info('Results saved to %s', args.result_dir)
    except StopAfterStage as exc:
        logger.info('Stopped after requested stage: %s', exc)
    finally:
        cleanup_offload_root(args)


if __name__ == '__main__':
    raise SystemExit(main())
