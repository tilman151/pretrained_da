from run_daan import run_multiple as run_daan
from run_pretraining import run_multiple as run_pretraining


def run(datasets, broken, domain_tradeoff, record_embeddings, pretraining_reps, replications, gpu):
    pretrained_checkpoints = run_pretraining(*datasets, broken, [0.01], record_embeddings, pretraining_reps, gpu)
    for b, checkpoint_per_tradeoff in pretrained_checkpoints.items():
        for _, checkpoints in checkpoint_per_tradeoff.items():
            for pretrained_checkpoint in checkpoints:
                run_daan(datasets[0], datasets[1], [b], domain_tradeoff, False,
                         record_embeddings, replications, gpu, pretrained_checkpoint)
                run_daan(datasets[1], datasets[0], [b], domain_tradeoff, False,
                         record_embeddings, replications, gpu, pretrained_checkpoint)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run pretraining and domain adaption for one dataset pairing')
    parser.add_argument('--datasets', nargs=2, type=int, help='a pair of FD numbers')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='*', type=float, help='tradeoff for domain classification')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('-p', '--pretraining_reps', type=int, default=1, help='replications for each pretraining run')
    parser.add_argument('-r', '--replications', type=int, default=10, help='replications for each adaption run')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run(opt.datasets, opt.broken, opt.domain_tradeoff,
        opt.record_embeddings, opt.pretraining_reps, opt.replications, opt.gpu)
