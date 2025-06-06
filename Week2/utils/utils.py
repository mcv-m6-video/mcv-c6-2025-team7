import torch
import torchvision.transforms as transforms

def get_optimizer(params, model):
    if params['optimizer'] == 'adam':
        #optimizer = Adam(learning_rate=float(params['lr']), beta_1=float(params['momentum']))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params['lr']))
    elif params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        #optimizer = SGD(learning_rate=float(params['lr']), momentum=float(params['momentum']))
        optimizer = torch.optim.SGD(model.parameters(), lr = float(params['lr']))
    elif params['optimizer'] == 'rmsprop':
        #optimizer = RMSprop(learning_rate=float(params['lr']), rho=float(params['momentum']))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=float(params['lr']))

    return optimizer


def data_augmentation(augment: bool, params=None, width=224, height=224):
    """
    Generates augmented data.
    :param augment: Boolean, when true the data augmentation is done.
    :return: ImageDataGenerator object instance
    """
    rotation_range = params['rot']
    shear_range=params['sr']
    zoom_range=params['zr']
    horizontal_flip=params['hf']

    if augment:
        transform = transforms.Compose([
                    transforms.Resize((width, height)),
                    transforms.RandomHorizontalFlip(horizontal_flip),
                    transforms.RandomRotation(rotation_range),
                    transforms.RandomAffine(degrees=0, shear=(-shear_range, shear_range)), #Shear
                    transforms.RandomResizedCrop(size=(width, height), scale=(1.0-zoom_range, 1.0)), #Zoom
                    transforms.ToTensor(),
                    transforms.Normalize(
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )])

        return transform

    else:
        transform = transforms.Compose([
                    transforms.Resize((width, height)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )])

        return transform
    