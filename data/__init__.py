from data.dataset_type import DatasetType
from data import femnist, mnist, chexpert, cifar100, cxr, fashion_mnist, celeba, synthetic, poets#, cass_retro


def get_dataset_instance(dataset_cfg, centralised=False, log_sample_data=False):
    dataset_type = DatasetType.from_value(dataset_cfg.name)

    if dataset_type == DatasetType.MNIST:
        dataset_class = mnist.MNISTDataset
    elif dataset_type in [DatasetType.FMNIST, DatasetType.FEMNIST]:
        dataset_class = femnist.FEMNISTDataset
    elif dataset_type == DatasetType.CIFAR100:
        dataset_class = cifar100.CIFAR100Dataset
    elif dataset_type == DatasetType.CHEXPERT:
        dataset_class = chexpert.CheXpertDataset
    elif dataset_type == DatasetType.CXR:
        dataset_class = cxr.CXRDataset
    elif dataset_type == DatasetType.FASHION_MNIST:
        dataset_class = fashion_mnist.FashionMNISTDataset
    elif dataset_type == DatasetType.CELEBA:
        dataset_class = celeba.CelebADataset
    elif dataset_type == DatasetType.SYNTHETIC:
        dataset_class = synthetic.SyntheticDataset
    elif dataset_type == DatasetType.POETS:
        dataset_class = poets.PoetsDataset
    #elif dataset_type == DatasetType.CASS_RETRO:
    #    dataset_class = cass_retro.CassRetroDataset
    else:
        raise NotImplementedError('Dataset type not defined')

    dataset = dataset_class(dataset_cfg, normalisation_mean_zero=False, centralised=centralised)

    if log_sample_data:
        dataset.log_sample_data()
    return dataset
