import dataset as ds
import torch
import torchvision


if __name__ == '__main__':

    dataset = ds.ContainersDataset(dataset_path='/home/home/PycharmProjects/datasets/containers', split='test')
    DL_test = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )

    num_classes = 3
    net_size = 18
    opt = 'SGD'
    epoch2load = 20

    if net_size == 18:
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif net_size == 34:
        model = torchvision.models.resnet34(num_classes=num_classes)
    elif net_size == 50:
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif net_size == 101:
        model = torchvision.models.resnet101(num_classes=num_classes)
    else:
        raise NotImplementedError
    print(f'[+] Selected model: Resnet{net_size}')

    state_dict = torch.load(f'./checkpoints/Resnet{net_size}_{opt}_Epoch_{epoch2load}.pth')
    model.load_state_dict(state_dict=state_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    target = 0
    for data in DL_test:
        imgs, labels = data[0], data[1]
        imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            result = model(imgs)
        _, predictions = torch.max(result, dim=1)
        correct_ = predictions.eq(labels).sum().float()
        target += correct_

    rez = target / DL_test.__len__()
    print(f"{target}/{DL_test.__len__()} ---> Accuracy = {rez}")

