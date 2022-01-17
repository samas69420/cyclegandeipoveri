from dataset import *
from models import *
from nets import *
import utils
import os

def train():
    ds = CustomLoader() # al posto di create_dataset
    model = CycleGANModel()
    model.setup()
    totiters = 0

    # per convertire un modello dataparallel in uno normale
    #model.netG_A.to(0)
    #model.netG_B.to(0)
    #model.netD_A.to(0)
    #model.netD_B.to(0)

    #save_net(model.netG_A, "G_A")
    #save_net(model.netG_B, "G_B")
    #save_net(model.netD_A, "D_A")
    #save_net(model.netD_B, "D_B")
    #quit()

    if not load:
        with torch.no_grad():
            trueB = ds.dataset.transform(Image.open(test_pic_path).convert('RGB')).to(model.device)
            trueB = trueB.view(-1,3,256,256)
            fakeA = model.netG_B(trueB)  # G_B(B)
            save_image(tensor2im(fakeA),f"{progressive_test_dir}/test_tr_fakeA_{totiters}.jpg")   
    elif load:
        totiters = utils.get_last_iter()
        print(f"continuing from iter: {totiters}")

    for epoch in range(n_epochs):
        for i,data in enumerate(ds):
            model.set_input(data)
            model.optimize_parameters()
            totiters += 1
            print(f"iters: {totiters}")
            if totiters % save_freq == 0:
                save_image(tensor2im(model.fake_B),f"{savedir}/pics/fakeb_{totiters}.jpg")   
                save_image(tensor2im(model.fake_A),f"{savedir}/pics/fakea_{totiters}.jpg")   
                save_net(model.netG_A, "G_A")
                save_net(model.netG_B, "G_B")
                save_net(model.netD_A, "D_A")
                save_net(model.netD_B, "D_B")
                with torch.no_grad():
                    #trueA = ds.dataset.transform(Image.open("testA.jpg").convert('RGB')).to(model.device)
                    #trueA = trueA.view(-1,3,256,256)
                    trueB = ds.dataset.transform(Image.open(test_pic_path).convert('RGB')).to(model.device)
                    trueB = trueB.view(-1,3,256,256)
                    #fakeB = model.netG_A(trueA)  # G_A(A)
                    fakeA = model.netG_B(trueB)  # G_B(B)
                    #save_image(tensor2im(fakeB),"test_tr_fakeB.jpg")   
                    save_image(tensor2im(fakeA),f"{progressive_test_dir}/test_tr_fakeA_{totiters}.jpg")   


def test():
    transform = get_transform()
    #trueA = transform(Image.open("testA.jpg").convert('RGB')).cuda()
    trueB = transform(Image.open("testB.jpg").convert('RGB')).cuda()
    #trueA = trueA.view(-1,3,256,256)
    trueB = trueB.view(-1,3,256,256)
    #netG_A = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, [0], True, "G_A")
    #fake_B = netG_A(trueA)
    netG_B = define_G(3, 3, 64, 'resnet_9blocks', 'instance', True, 'normal', 0.02, [0], True, "G_B")
    fake_A = netG_B(trueB)
    #save_image(tensor2im(fake_B),"test_fakeB.jpg")   
    save_image(tensor2im(fake_A),"test_fakeA.jpg")   

if __name__ == "__main__":
    train()
    #test()
