import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = ['trainloader','testloader']


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



def trainloader(args):

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='../data/CIFAR10/',
                                train=True, 
                                transform=transform_train,
                                download=True)
        
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='../data/CIFAR100/',
                                train=True, 
                                transform=transform_train,
                                download=True)
        

    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)
    
    print(f'number of train data: {len(train_loader)}')
    return train_loader



def testloader(args):
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='../data/CIFAR10/',
                                        train=False, 
                                        transform=transform_test)
        
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root='../data/CIFAR100/',
                                        train=False, 
                                        transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False,
                                              num_workers=args.num_workers)
    
    print(f'number of test data: {len(test_loader)}')
    return test_loader


