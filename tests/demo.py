import sys

from numba.tests.test_random import numpy_seed

sys.path.append('..')

from src.dpe import DPE


def main():
    dpe = DPE(
        data_dir=r'/scratch/ssd004/scratch/minht/checkpoints/sd0/Waterbirds/13574640',
        metadata_path='/scratch/ssd004/scratch/minht/datasets/waterbirds/metadata_waterbirds.csv',
        verbose=True,
        num_stages=5,
        seed=0,
        device='cuda',
    )
    dpe.fit()
    print('Done')


if __name__ == '__main__':
    main()