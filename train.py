import os.path

import dataset as ds
import torch
import torchvision

if __name__ == '__main__':

    num_classes = 3
    net_size = 18
    opt = 'SGD'
    learning_rate = 0.1
    epochs2change_lr = 5
    load_pretrained = True
    epoch2load = 8
    weight_decay_coefficient = 0.1

    dataset = ds.ContainersDataset(dataset_path='/home/home/PycharmProjects/datasets/containers', split='train')
    DL_train = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=20,
        shuffle=True
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

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

    if opt == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    elif opt == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError
    print(f'[+] Selected optimizer: {opt}')

    checkpoint_path = f'./checkpoints/Resnet{net_size}_{opt}_Epoch_{epoch2load}.pth'
    if load_pretrained and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict=state_dict)
        print(f'[+] Pretrained weights from {checkpoint_path} were loaded.')
    else:
        print('[!] - No pretrained weights.')

    if torch.cuda.is_available():
        print('[*] - Moving model to device.')
        model = model.cuda()

    for epoch in range(50):
        cumulative_loss = 0
        print(f'[*] Current Learning rate - {learning_rate}')
        for i, (imgs, lbls) in enumerate(DL_train):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                lbls = lbls.cuda()
            optimizer.zero_grad()
            result = model(imgs)

            loss = loss_fn(result, lbls)
            cumulative_loss += loss.item()
            if i != 0 and i % 20 == 0:
                print(f"[*] -> Epoch: {epoch} - Iter: {i} - Loss: {round(cumulative_loss / (i + 1), 3)}")
            loss.backward()
            optimizer.step()

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        torch.save(model.state_dict(), f'./checkpoints/Resnet{net_size}_{opt}_Epoch_{epoch}.pth')
        print(f'[+] Trained weights saved to ./checkpoints/Resnet{net_size}_{opt}_Epoch_{epoch}.pth')

        if epoch != 0 and epoch % epochs2change_lr == 0:
            learning_rate *= weight_decay_coefficient
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * weight_decay_coefficient
