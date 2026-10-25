"""Sequential resident manifests for one uninterrupted optimizer/RNG stream.

Chunks run in CLI order every epoch. Each uses the legacy manifest shuffle,
including its initial full-row randperm when a keep mask is present. Checkpoints
store only cursors, losses and RNG states, never the shuffled row vector.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch

try:
    from .nnue import eval_to_winrate
except ImportError:
    from nnue import eval_to_winrate


def rng_state(device):
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def chunk_targets(manifest, args):
    """Same raw-label/result transforms as train_nnue's manifest path (CPU)."""
    evals = torch.from_numpy(manifest.vector('labels'))
    need_result = (args.wdl_weight > 0 or
                   (args.objective == 'wdl' and args.wdl_lambda < 1) or
                   (args.objective == 'cp' and args.cp_result_lambda < 1))
    result = torch.from_numpy((manifest.vector('wdl') + 1.0) / 2.0) if need_result else None
    if args.objective == 'cp':
        target = evals.clamp(-args.clamp_cp, args.clamp_cp)
        if args.cp_result_lambda < 1:
            known = result != 0.5
            mixed = (args.cp_result_lambda * eval_to_winrate(evals, args.scale_cp) +
                     (1.0 - args.cp_result_lambda) * result)
            mixed_cp = args.scale_cp * torch.logit(mixed.clamp(1e-6, 1 - 1e-6))
            target[known] = mixed_cp[known].clamp(-args.clamp_cp, args.clamp_cp)
    elif args.objective == 'wdl':
        lam = args.wdl_lambda
        target = lam * eval_to_winrate(evals, args.scale_cp)
        if lam < 1:
            if args.wdl_known_only:
                mix = (1.0 - lam) * (result != 0.5).float()
                target = (1.0 - mix) * eval_to_winrate(evals, args.scale_cp) + mix * result
            else:
                target = target + (1.0 - lam) * result
    else:
        target = eval_to_winrate(evals, args.scale_cp)
    return target, result


def train_epoch(manifests, *, args, epoch, model, optimizer, generator, device,
                criterion, mirror_perm, run_batches, checkpoint, paused,
                updates, resume=None):
    """Train one epoch, releasing every resident chunk before loading the next.

    checkpoint(state, updates) uses train_nnue's atomic checkpoint writer. A state
    at chunk_index == len(manifests), or at the update cap, resumes validation
    without making an extra optimizer update.
    """
    saved = resume or {}
    completed = list(saved.get('completed', []))
    first = saved.get('chunk_index', 0)
    prior_secs = saved.get('epoch_secs', 0.0)
    started = time.time()
    last_ckpt = time.time()
    partial = saved.get('partial', False)

    def progress(index, cursor=0, order_state=None, loss=0.0):
        return dict(chunk_index=index, cursor=cursor, epoch_generator=order_state,
                    loss_so_far=float(loss), completed=list(completed), partial=partial,
                    epoch_secs=prior_secs + time.time() - started)

    for index in range(first, len(manifests)):
        if args.max_steps is not None and updates >= args.max_steps:
            break
        manifest = manifests[index]
        # No old indices/targets/order remain in this frame at the next load.
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        indices = manifest.resident(device)
        target, result = chunk_targets(manifest, args)
        target = target.to(device)
        wdl = result.to(device) if result is not None else None
        mask = manifest.vector('mask')
        keep = torch.from_numpy(np.flatnonzero(mask).astype(np.int64)) if not mask.all() else None
        if not mask.any():
            raise ValueError(f'chunk {index}: empty training mask')

        def order_for_chunk():
            order = torch.randperm(manifest.rows, generator=generator)
            if keep is not None:
                order = keep[torch.randperm(len(keep), generator=generator)]
            return order

        if index == first and saved.get('cursor', 0) > 0:
            current_gen = generator.get_state()
            order_state = saved['epoch_generator']
            generator.set_state(order_state)
            order = order_for_chunk()
            generator.set_state(current_gen)
            cursor, loss_so_far = saved['cursor'], saved['loss_so_far']
        else:
            order_state = generator.get_state()
            order = order_for_chunk()
            cursor, loss_so_far = 0, 0.0
        end = (min(len(order), cursor + (args.max_steps - updates) * args.batch_size)
               if args.max_steps is not None else len(order))

        def on_batch(next_start, loss):
            nonlocal updates, last_ckpt
            updates += 1
            stop = args.stop_file is not None and os.path.exists(args.stop_file)
            due = args.checkpoint_minutes > 0 and time.time() - last_ckpt >= 60 * args.checkpoint_minutes
            if (stop or due) and next_start < end:
                checkpoint(progress(index, next_start, order_state, loss), updates)
                last_ckpt = time.time()
                print(f'checkpoint epoch={epoch} chunk={index + 1} cursor={next_start}/{len(order)}', flush=True)
                if stop:
                    raise paused()

        loss, _ = run_batches(
            model, indices, target, args.batch_size, 77, order, train=True,
            objective=args.objective, criterion=criterion, scale_cp=args.scale_cp,
            optimizer=optimizer, mirror_perm=mirror_perm, gen=generator, device=device,
            canonical=args.canonical, shared=True, wdl_target=wdl, wdl_weight=args.wdl_weight,
            start_at=cursor, loss_so_far=loss_so_far, on_batch=on_batch,
            async_loss=args.async_loss, end_at=end)
        entry = dict(chunk=index, rows=end, train_loss=loss)
        if device == 'cuda':
            torch.cuda.synchronize()
            entry.update(peak_allocated=torch.cuda.max_memory_allocated(),
                         peak_reserved=torch.cuda.max_memory_reserved())
            print(f'CUDA_CHUNK epoch={epoch} chunk={index + 1} '
                  f'allocated={entry["peak_allocated"]} reserved={entry["peak_reserved"]}', flush=True)
        print(f'epoch={epoch} chunk={index + 1}/{len(manifests)} rows={end} train_loss={loss:.8g}', flush=True)
        completed.append(entry)
        partial = end < len(order) or index + 1 < len(manifests)
        del indices, target, result, wdl, order, keep, mask
        if device == 'cuda':
            torch.cuda.empty_cache()
        # Always publish a chunk boundary when checkpointing is enabled, even if
        # the interval has not elapsed. Validation/scheduler have not run yet.
        if args.checkpoint_minutes > 0 or args.resume or args.stop_file:
            checkpoint(progress(index + 1), updates)
            last_ckpt = time.time()
        if args.stop_file and os.path.exists(args.stop_file):
            raise paused()

    rows = sum(c['rows'] for c in completed)
    loss = sum(c['train_loss'] * c['rows'] for c in completed) / rows
    return loss, prior_secs + time.time() - started, completed, partial, updates
