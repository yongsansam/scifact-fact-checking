# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.label_scoring import BINARY_LABELS, LABELS, score_candidate_labels


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    if opt.balance_labels:
        counts = {}
        for example in train_dataset.data:
            label = example.get('target', example.get('answers', [''])[0])
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[example.get('target', example.get('answers', [''])[0])]
                   for example in train_dataset.data]
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    # `step` stored in checkpoints is an optimizer-update count.  A previous
    # version incremented it for every micro-batch, which made 1000 requested
    # steps equal only 125 optimizer updates when accumulation_steps=8.
    update = step
    micro_step = 0
    curr_loss = 0.0
    curr_micro_batches = 0
    epoch = 1
    model.zero_grad()
    model.train()
    while update < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            # Match the effective gradient scale used by the controlled
            # baseline: average, rather than sum, accumulated micro-batches.
            (train_loss / opt.accumulation_steps).backward()

            reported_loss = src.util.average_main(train_loss.detach(), opt)
            curr_loss += reported_loss.item()
            curr_micro_batches += 1
            micro_step += 1

            if micro_step % opt.accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            update += 1

            if update % opt.eval_freq == 0 or update == opt.total_steps:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, update, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    mean_train_loss = curr_loss / max(curr_micro_batches, 1)
                    log = f"{update} / {opt.total_steps} optimizer updates |"
                    log += f"train: {mean_train_loss:.3f} |"
                    metric_name = 'macro-F1' if opt.selection_metric == 'macro_f1' else 'accuracy'
                    log += f"evaluation: {100*dev_em:.2f}{metric_name} |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, update)
                        tb_logger.add_scalar("Training", mean_train_loss, update)
                    curr_loss = 0.
                    curr_micro_batches = 0

            if opt.is_main and update % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, update, best_dev_em,
                          opt, checkpoint_path, f"step-{update}")
            if update >= opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    active_labels = BINARY_LABELS if opt.binary_labels else LABELS
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    predictions = []
    targets = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            scores = score_candidate_labels(
                model,
                tokenizer,
                context_ids.cuda(),
                context_mask.cuda(),
                reduction='sum',
                labels=active_labels,
            )
            batch_predictions = [active_labels[index] for index in scores.argmax(dim=1).tolist()]
            for k, prediction in enumerate(batch_predictions):
                gold = dataset.get_example(idx[k])['answers'][0].strip().upper()
                total += 1
                predictions.append(prediction)
                targets.append(gold)

    accuracy = sum(p == y for p, y in zip(predictions, targets)) / max(total, 1)
    if opt.selection_metric == 'accuracy':
        return accuracy
    f1_values = []
    for label in active_labels:
        tp = sum(p == label and y == label for p, y in zip(predictions, targets))
        fp = sum(p == label and y != label for p, y in zip(predictions, targets))
        fn = sum(p != label and y == label for p, y in zip(predictions, targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values)

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir).expanduser().resolve() / opt.name
    latest_path = checkpoint_path / "checkpoint" / "latest"

    checkpoint_exists = (
    latest_path.exists()
    and (latest_path / "config.json").exists()
    )
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = opt.model_name or ('t5-' + opt.model_size)
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
