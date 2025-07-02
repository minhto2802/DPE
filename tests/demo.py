import sys
sys.path.append('..')

from src.dpe import DPE
from src.dpe.misc import fix_random_seed


def main():
    SEED = 0
    fix_random_seed(SEED)

    dpe = DPE(
        data_dir=r'/scratch/ssd004/scratch/minht/checkpoints/sd0/Waterbirds/13574640',
        metadata_path='/scratch/ssd004/scratch/minht/datasets/waterbirds/metadata_waterbirds.csv',
        verbose=True,
        num_stages=5,
        seed=0,
        device='cuda',
    )
    dpe.fit()

if __name__ == '__main__':
    main()