#todo add loss check on validation set

import torchvision.models as models
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#how many classes (for this model its 2: hotdog and not-hotdog)
classes_n=2
#should use cuda?
use_cuda=False

#get pretrained vgg16 model
def get_pretrained():
	return models.vgg16(pretrained=True)

#replace last layer
def prepare_for_finetunning(model):

    for param in model.parameters():
        param.requires_grad = False
        param.requires_grad = True

    #replacing last layer with new fully connected
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(torch.nn.Linear(4096, classes_n))
    new_classifier = torch.nn.Sequential(*mod)
    model.classifier = new_classifier
    return

#finetune and save model => main function
def finetune_and_save(path):
    model=get_pretrained()
    prepare_for_finetunning(model)
    print("start finetunning")
    #resize image and convert to tensor. need to be 224 to use vgg16 or other pretrained network
    trans=transforms.Compose([transforms.Scale(400),transforms.CenterCrop(size=[224,224]),transforms.ToTensor(),])
    #create data loader
    data = ImageFolder(root='train_set', transform=trans)
    data_loader = DataLoader(data, shuffle=True, batch_size=1)
    #train
    for epoch in range(1):
        finetune(model, data_loader, epoch)
    print("saving model")
    #save
    torch.save(model, path)
    return

#finetune model
def finetune(model, train_loader, epoch):
    #get last layer
    classifierLayers=list(model.classifier.children())
    params=classifierLayers[len(classifierLayers)-1].parameters()
    #optimize only last layer
    optimizer=optim.Adam(params=params, lr=0.01)
    #set to train mode
    model.train()

    for batch_i, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss=F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_i % 5 == 0:
            print('epoch: {}, epoch progress: ({}/{} ({:.0f}%)) \tloss: {:.6f}'
                  .format(epoch, batch_i * len(data), len(train_loader.dataset),
                          100. * batch_i / len(train_loader), loss.data[0]))


#finetune model just by running this script
finetune_and_save("finetunned_model")
