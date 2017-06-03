import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# desired size of the output image, need to be 224,244 to use pretrained vgg16
imgsize = 224,224

#transform PIL image to tensor
def image_to_tensor(pil_image):
    #resize image
    resized=ImageOps.fit(pil_image, imgsize, Image.ANTIALIAS)
    # transform it into a torch tensor
    loader = transforms.Compose([
        transforms.ToTensor()])
    return loader(resized).unsqueeze(0) #need to add one dimension, need to be 4D to pass into the network

#load model
def load_model():
   return torch.load("finetunned_model")

#get label for image at path "path"
def is_hotdog(path):
    one_image = load_image(path)
    image_tensor = image_to_tensor(one_image)
    image_as_variable = Variable(image_tensor)
    model = load_model()
    model.eval()
    probabilities = model.forward(image_as_variable)
    print(probabilities)
    hotdog_prob=get_hotdog_probability(probabilities)
    print("hotdog probability: {}%".format(hotdog_prob*100))
    return hotdog_prob>0.7

#all_probabilities is a Variable
def get_hotdog_probability(all_probabilities):
    probs=all_probabilities.data.numpy()[0]
    return probs[0]>probs[1]

#load image from path as PIL image
def load_image(path):
    return Image.open(path)
