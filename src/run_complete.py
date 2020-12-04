from run_daan import run_multiple as run_daan
from run_pretraining import run_multiple as run_pretraining


def run(source, target, percent_broken, domain_tradeoff, record_embeddings, pretraining_reps, replications, gpu):
    pretrained_checkpoints = run_pretraining(source, target, percent_broken, domain_tradeoff,
                                             record_embeddings, pretraining_reps, gpu)
    for broken, checkpoint_per_tradeoff in pretrained_checkpoints.items():
        for _, checkpoints in checkpoint_per_tradeoff.items():
            for pretrained_checkpoint in checkpoints:
                run_daan(source, target, [broken], [1.0], record_embeddings, replications, gpu, pretrained_checkpoint)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run pretraining and domain adaption for one dataset pairing')
    parser.add_argument('--source', required=True, type=int, help='FD number of the source data')
    parser.add_argument('--target', required=True, type=int, help='FD number of the target data')
    parser.add_argument('-b', '--broken', nargs='*', type=float, help='percent broken to use')
    parser.add_argument('--domain_tradeoff', nargs='*', type=float, default=[0.01], help='tradeoff for pre-training')
    parser.add_argument('-p', '--pretraining_reps', type=int, default=1, help='replications for each pretraining run')
    parser.add_argument('-r', '--replications', type=int, default=10, help='replications for each adaption run')
    parser.add_argument('--record_embeddings', action='store_true', help='whether to record embeddings of val data')
    parser.add_argument('--gpu', type=int, default=0, help='id of GPU to use')
    opt = parser.parse_args()

    run(opt.source, opt.target, opt.broken, opt.domain_tradeoff,
        opt.record_embeddings, opt.pretraining_reps, opt.replications, opt.gpu)
