import numpy as np
import os
import functools
import itertools
import random
import torch
from torch.nn import init
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image as torch_save_image
from abc import ABC, abstractmethod

# DATASET E PREPROCESSING

class BaseDataset(data.Dataset, ABC):
    def __init__(self):
        pass

class UnalignedDataset(BaseDataset):
    def __init__(self):
        BaseDataset.__init__(self)
        #self.dir_A = "datasets/monet2photo/trainA"
        #self.dir_B = "datasets/monet2photo/trainB" 
        self.dir_A = f"{datasetpath}/trainA"
        self.dir_B = f"{datasetpath}/trainB" 
        self.A_paths = sorted(make_dataset(self.dir_A))   
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  
        input_nc = 3
        output_nc = 3
        self.transform = get_transform()
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size] 
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img) # a e b vengono trasformati in tensori da transform
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    def __len__(self):
        return max(self.A_size, self.B_size)

class CustomLoader():
    def __init__(self):
        self.dataset = UnalignedDataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = batchsize,
            shuffle = False, # con batch size 1 dovrebbe non servire a niente questo
            num_workers = 1 # num threads
        )
    def __iter__(self): # quindi iterare su customloader dovrebbe essere come iterare su dataloader?
        for data in self.dataloader:
            yield data
    def __len__(self):
        return len(self.dataset)

def make_dataset(dir):
    paths = []
    for root,subdirs,elements in os.walk(dir):
        for element in elements:
            if element.endswith(".jpg"):
                paths.append(root+'/'+element) 
    return paths

def get_transform(): # no_flip = false params = None convert = True grayscale = False
    method=Image.BICUBIC #originale ma mi da warning
    #method = transforms.InterpolationMode.BICUBIC # dovrebbe fare la stessa cosa di image.bicubic ma senza warning
    transform_list = []
    transform_list.append(transforms.Resize([256,256], method))
    transform_list.append(transforms.RandomCrop(256))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# MODEL

class BaseModel(ABC): 
    def __init__(self): # istrain = True gpu_ids = 0 
        self.device = torch.device('cuda:0') # cuda:0
        self.save_dir = os.path.join(savedir)  # save all the checkpoints to save_dir
        torch.backends.cudnn.benchmark = True # a che serve?
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
    def get_image_paths(self):
        return self.image_paths
    def setup(self):
        self.print_networks() 
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
class CycleGANModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.gpu_ids = [0]
        self.netG_A = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, self.gpu_ids, load, "G_A")
        self.netG_B = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, self.gpu_ids, load, "G_B")
        self.netD_A = define_D(3, 64, 'basic', 3, 'instance', 'normal', 0.02, self.gpu_ids, load, "D_A")
        self.netD_B = define_D(3, 64, 'basic', 3, 'instance', 'normal', 0.02, self.gpu_ids, load, "D_B")
        self.fake_A_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(50)  # create image buffer to store previously generated images
        #loss functions
        self.criterionGAN = GANLoss().to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        #initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
    def optimize_parameters(self):
        self.forward() 
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        #self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        #self.set_requires_grad([self.netG_A, self.netG_B], False)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    def backward_G(self):
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    

def define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type='normal', init_gain=0.02, gpu_ids=[], load=False, name = None):
    net = None
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    return init_net(net, init_type, init_gain, gpu_ids, load, name)
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type='reflect', use_bias=True):
        super(ResnetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                      kernel_size=3, stride=2,
                      padding=1, output_padding=1,
                      bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        #conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x) 
        return out

def init_net(net, init_type, init_gain, gpu_ids, load = False, name = None):
    net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs non ho capito ma a che serve? se ho capito bene ricopia il modulo in ogni device e fa tipo albero con i gradienti
    if not load:
        init_weights(net, init_type, init_gain=init_gain)
    elif load:
        net.load_state_dict(torch.load(f"{savedir}/{name}.rete"))
    return net
def init_weights(net, init_type, init_gain):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): # se m è un layer di convoluzione o lineare
            init.normal_(m.weight.data, 0.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies. # se invece m è un layer di normalizzazione
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>

def define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpu_ids=[], load = False, name = None):
    net = None
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = NLayerDiscriminator(norm_layer=norm_layer, input_nc=3, ndf=64, n_layers=3 )
    return init_net(net, init_type, init_gain, gpu_ids, load, name)
class NLayerDiscriminator(nn.Module):
    def __init__(self, norm_layer, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        use_bias = True
        kw = 4
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        return self.model(input)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  
            self.num_imgs = 0
            self.images = []
    def query(self, images):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer rimpiazzando quella scelta
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.loss = nn.MSELoss()
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)



def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_net(net,name):
    torch.save(net.state_dict(),f"{savedir}/{name}.rete") 
    #torch.save(net,f"{savedir}/{name}.rete")

def load_net(name):
    #model = nn.DataParallel(nn.Module)
    #model.load_state_dict(torch.load(f"{savedir}/{name}.rete"))
    model = torch.load(f"{savedir}/{name}.rete")
    return model



# general settings
n_epochs = 200
lr = 0.0002
batchsize = 1
savedir = "./salvataggi"
datasetpath = "./datasets/synth"
lambda_A = 10.0
lambda_B = 10.0
lambda_idt = 0.5
load = True
#load = False

def train():
    #net = load_net("G_A")
    #print(net)
    ds = CustomLoader() # al posto di create_dataset
    model = CycleGANModel()
    model.setup()
    totiters = 0
    for epoch in range(n_epochs):
        for i,data in enumerate(ds):
            model.set_input(data)
            model.optimize_parameters()
            totiters += 1
            print(f"epoch: {epoch} iters: {totiters}")
            if totiters % 100 == 0:
                #torch_save_image(model.fake_B,f"{savedir}/torchpics/fakeb_{totiters}.png")   
                #torch_save_image(model.fake_A,f"{savedir}/torchpics/fakea_{totiters}.png")   
                save_image(tensor2im(model.fake_B),f"{savedir}/pics/fakeb_{totiters}.png")   
                save_image(tensor2im(model.fake_A),f"{savedir}/pics/fakea_{totiters}.png")   
                save_net(model.netG_A, "G_A")
                save_net(model.netG_B, "G_B")
                save_net(model.netD_A, "D_A")
                save_net(model.netD_B, "D_B")
                with torch.no_grad():
                    #trueA = ds.dataset.transform(Image.open("testA.jpg").convert('RGB')).to(model.device)
                    #print(trueA.shape)
                    #trueA = trueA.view(-1,3,256,256)
                    #print(trueA.shape)
                    #quit()
                    trueB = ds.dataset.transform(Image.open("testB.jpg").convert('RGB')).to(model.device)
                    trueB = trueB.view(-1,3,256,256)
                    #fakeB = model.netG_A(trueA)  # G_A(A)
                    fakeA = model.netG_B(trueB)  # G_B(B)
                    #save_image(tensor2im(fakeB),"test_tr_fakeB.jpg")   
                    save_image(tensor2im(fakeA),"test_tr_fakeA.jpg")   


def test():
    transform = get_transform()
    trueA = transform(Image.open("testA.jpg").convert('RGB')).cuda()
    trueB = transform(Image.open("testB.jpg").convert('RGB')).cuda()
    trueA = trueA.view(-1,3,256,256)
    trueB = trueB.view(-1,3,256,256)
    netG_A = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, [0], True, "G_A")
    fake_B = netG_A(trueA)
    netG_B = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, [0], True, "G_B")
    fake_A = netG_B(trueB)
    save_image(tensor2im(fake_B),"test_fakeB.jpg")   
    save_image(tensor2im(fake_A),"test_fakeA.jpg")   

if __name__ == "__main__":
    train()
    #test()
