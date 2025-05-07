from enum import Enum


class DatasetType(Enum):
    CXR = "CXR"
    CHEXPERT = "CheXpert"
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    FEMNIST = "FEMNIST"
    CIFAR100 = "CIFAR100"
    FASHION_MNIST = "FashionMNIST"
    CELEBA = "CelebA"
    CASS_RETRO = "CassRetro"
    SYNTHETIC = "Synthetic"
    POETS = "Poets"

    @classmethod
    def from_value(cls, value):
        try:
            return cls(value)
        except ValueError:
            valid_values = [d.value for d in DatasetType]
            raise ValueError(f"The dataset {value} is not supported. Use one of {', '.join(valid_values)}")
