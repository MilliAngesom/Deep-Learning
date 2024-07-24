from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            for m in remove_class:
                mask = self.targets != m #remove_class
                self.images = self.images[mask]
                self.targets = self.targets[mask]
                self.classes.remove(m)#(remove_class)

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]



    def _sample_positive(self, index):
        
        # YOUR CODE HERE

        target_id = self.targets[index].item()
        positive_indices = self.target2indices[target_id]
        positive_indices = [i for i in positive_indices if i != index]  # Exclude the anchor index
        return choice(positive_indices)

    def _sample_negative(self, index):

        # YOUR CODE HERE

        target_id = self.targets[index].item()
        negative_classes = [cls for cls in self.classes if cls != target_id]
        negative_class = choice(negative_classes)
        return choice(self.target2indices[negative_class])



    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)