# by yi 数据集
from config import *
from datasets.data_loader import *
from easydl import *
from torchvision.transforms.transforms import *

source_classes = [i for i in range(args.datasets.n_classes)]
target_classes = [i for i in range(args.datasets.n_classes)]
target_aux_classes = [i for i in range(args.datasets.aux_classes + 1)]

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
])


test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
])

# by yi self supervision target data and test data
target_train_dl = get_target_dataloader(args.datasets.tar)  # 训练数据集
target_pseudo_label_dl = get_pseudo_label_dataloader(args.datasets.tar)  # 带旋转标签的数据集
target_test_dl = get_test_dataloader(args.datasets.test) # 测试数据集
_, target_val_loader = get_train_val_dataloader(args.datasets.tar)  # 验证集数据，main中使用其作为测试集数据
