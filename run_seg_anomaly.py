import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    datasets = ['mvtec_dtd']
    shots = [1]
    aa = [0.1, 0.3, 0.5, 0.7, 0.9]

    for shot in shots:
        for a in aa:
            for dataset in datasets:
                classes = dataset_classes[dataset]
                for cls in classes[:]:
                    sh_method = f'python train_seg_anomaly.py ' \
                                f'--dataset {dataset} ' \
                                f'--k-shot {shot} ' \
                                f'--class_name {cls} ' \
                                f'--a {a} ' \

                    print(sh_method)
                    pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




