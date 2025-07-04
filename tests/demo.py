import sys
sys.path.append('..')

from dpe import DPE


def main():
    dpe = DPE(
        data_dir=r'/scratch/ssd004/scratch/minht/checkpoints/sd0/Waterbirds/13574640',
        metadata_path='/scratch/ssd004/scratch/minht/datasets/waterbirds/metadata_waterbirds.csv',
        num_stages=2,
        device='cuda',
        eval_freq=1,
        train_attr='no',
        seed=0,
    )
    dpe.fit()
    print('Demo completed successfully!')

if __name__ == '__main__':
    main()
