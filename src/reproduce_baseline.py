import os
from datetime import datetime

from run_semi_supervised import run


script_path = os.path.dirname(__file__)
config_root = os.path.join(script_path, "..", "configs")


def reproduce(base_version):
    if base_version is None:
        base_version = datetime.now().timestamp()
    error_log = []
    fds = [1, 2, 3, 4]
    percent_fails = [1.0, 0.4, 0.2, 0.1, 0.02]

    for fd in fds:
        for fails in percent_fails:
            arch_config_path = os.path.join(config_root, f"baseline_fd{fd}.json")
            pre_config_path = os.path.join(config_root, f"baseline_pre_fd{fd}.json")
            try:
                run(
                    fd,
                    None,
                    fails,
                    arch_config_path,
                    pre_config_path,
                    pretrain=False,
                    mode="linear",
                    record_embeddings=False,
                    replications=10,
                    gpu=[0],
                    seeded=False,
                    version=f"{base_version}_baseline@{fails:.2f}",
                )
            except Exception as e:
                error_log.append(e)

    if error_log:
        for e in error_log:
            print(type(e), e)
    else:
        print("Everything went well.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproduce all supervised baseline experiments."
    )
    parser.add_argument(
        "base_version", default=None, help="common prefix for the version tag"
    )
    opt = parser.parse_args()

    reproduce(opt.base_version)
