# general settings

n_epochs = 200
lr = 0.0001
batchsize = 1
savedir = "../salvataggi"
test_pic_path = "../testB.jpg"
progressive_test_dir = f"{savedir}/pics/progressive"
datasetpath = "../datasets/synth"
save_freq = 500
lambda_A = 10.0
lambda_B = 10.0
lambda_idt = 0.5
load = True # when true load nets from savedir, when false save nets in savedir
#load = False
