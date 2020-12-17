import random

from run_dann import run as run_dann
from run_pretraining import run_multiple as run_pretraining


def run(source, target, percent_broken, domain_tradeoff, dropout,
        record_embeddings, pretraining_reps, best_only, gpu):
    pretrained_checkpoints = run_pretraining(source, target, percent_broken, domain_tradeoff, dropout,
                                             record_embeddings, pretraining_reps, gpu)
    random.seed(999)
    seeds = [random.randint(0, 9999999) for _ in range(pretraining_reps)]
    if best_only:
        _adapt_with_best_pretrained(source, target, pretrained_checkpoints, record_embeddings, seeds, gpu)
    else:
        _adapt_with_all_pretrained(source, target, pretrained_checkpoints, record_embeddings, seeds, gpu)


def _adapt_with_all_pretrained(source, target, pretrained_checkpoints, record_embeddings, seeds, gpu):
    for broken, checkpoint_per_tradeoff in pretrained_checkpoints.items():
        for checkpoints in checkpoint_per_tradeoff.values():
            for (pretrained_checkpoint, best_val_score), s in zip(checkpoints, seeds):
                run_dann(source, target, broken, 1.0, record_embeddings, s, gpu, pretrained_checkpoint)


def _adapt_with_best_pretrained(source, target, pretrained_checkpoints, record_embeddings, seeds, gpu):
    for broken, checkpoint_per_tradeoff in pretrained_checkpoints.items():
        for checkpoints in checkpoint_per_tradeoff.values():
            sorted_checkpoints = sorted(checkpoints, key=lambda x: x[1])  # sort by val loss
            best_pretrained_path = sorted_checkpoints[0][0]
            for s in seeds:
                run_dann(source, target, broken, 1.0, record_embeddings, s, gpu, best_pretrained_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run pretraining and domain adaption for one dataset pairing')
    parser.add_argument('--source', required=True, type=int, help='FD number of the source data')
    parser.add_argument('--target', required=True, type=int, help='FD number of the target data')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='*', type=float, default=[0.01], help='tradeoff for pre-training')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout used after each conv layer')
    parser.add_argument('-p', '--pretraining_reps', type=int, default=1, help='replications for each pretraining run')
    parser.add_argument('--best_only', action='store_true', help='adapt only on best pretraining run')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run(opt.source, opt.target, opt.broken, opt.domain_tradeoff, opt.dropout,
        opt.record_embeddings, opt.pretraining_reps, opt.best_only, opt.gpu)
