import torch
import os
import tqdm
import cv2
import numpy as np
import torch.nn as nn
from model import CVF_model
from loss import loss_aug, loss_main,ssim,psnr
from data import dataset
from torch.backends import cudnn

mse = nn.MSELoss(reduction='mean') 

def process_image(train_noisy):
    batch_size = train_noisy.size(0)  # 获取批量大小
    STD_train = []

    for h in range(3, train_noisy.size(2) - 3):
        for w in range(3, train_noisy.size(3) - 3):
            # 提取子图像并进行标准化
            sub_images = train_noisy[:, :, h-3:h+3, w-3:w+3] / 255.0
            
            # 计算子图像的标准差并添加到列表中
            std = torch.std(sub_images.view(batch_size, -1), dim=1, unbiased=False).view(-1, 1, 1)  # 使用无偏估计方法
            STD_train.append(std)

    # 计算所有标准差的平均值并返回
    return torch.mean(torch.cat(STD_train, dim=1), dim=1)

def save_image(opt,index,noise_output,clean_output):

    result_folder = './results/' + opt.name + '/test/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_folder += opt.model_name + '/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder + 'noise/'):
        os.makedirs(result_folder + 'noise/')
    if not os.path.exists(result_folder + 'clean/'):
        os.makedirs(result_folder + 'clean/')
    img = _save_image(opt,noise_output)
    cv2.imwrite(result_folder + 'noise/' + str(index) + '.bmp', img.astype(np.uint8))
    img = _save_image(opt,clean_output)
    cv2.imwrite(result_folder + 'clean/' + str(index) + '.bmp', img.astype(np.uint8))

def _save_image(opt,img):
    img = img.cpu().detach().numpy()
    img = img[:,:]
    img = img.reshape(opt.output_h,opt.output_w)
    img = img  * 255.0
    img = img.astype('uint8')
    return img 

def train(opt,model,device,train_dataloader,val_dataloader):
    epoch_num = opt.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-09, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if opt.Continue :
        epochs = int(opt.model_name) + 1
    else:
        epochs = 1
    
    for epoch in (range( epochs,epochs + epoch_num)):
        
        model.train()
        loss_train = 0.0
        
        for index, (noise_data,clear_data) in enumerate(tqdm.tqdm(train_dataloader)):

            noise_data = noise_data.float().to(device)
            clear_data = clear_data.float().to(device)
            
            std = process_image(noise_data)     
    
            optimizer.zero_grad()
            gamma = 1.0

            noise_w, noise_b, clean = model(noise_data)
            noise_w1, noise_b1, clean1 = model((clean))
            noise_w2, noise_b2, clean2 = model((clean+torch.pow(clean,gamma)*noise_w)) #1
            noise_w3, noise_b3, clean3 = model((noise_b))
            noise_w4, noise_b4, clean4 = model((clean+torch.pow(clean,gamma)*noise_w-noise_b)) #2
            noise_w5, noise_b5, clean5 = model((clean-torch.pow(clean,gamma)*noise_w+noise_b)) #3
            noise_w6, noise_b6, clean6 = model((clean-torch.pow(clean,gamma)*noise_w-noise_b)) #4
            noise_w10, noise_b10, clean10 = model((clean+torch.pow(clean,gamma)*noise_w+noise_b)) #5

            input_noisy_pred = clean+torch.pow(clean,gamma)*noise_w+noise_b
            
            loss_clear = mse(clean,clear_data)

            loss = loss_main(noise_data, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2,std,gamma)
            
            loss_neg1 = loss_aug(clean, clean4, noise_w, noise_w4, noise_b, -noise_b4)
            loss_neg2 = loss_aug(clean, clean5, noise_w, -noise_w5, noise_b, noise_b5)
            loss_neg3 = loss_aug(clean, clean6, noise_w, -noise_w6, noise_b, -noise_b6)
            loss_neg7 = loss_aug(clean, clean10, noise_w, noise_w10, noise_b, noise_b10)    

            loss_total = loss+.1*(loss_neg1+loss_neg2+loss_neg3+loss_neg7)+ loss_clear * 0.4
            
            loss_total.backward()
            optimizer.step()
            loss_train += loss_total.item()
        
        scheduler.step()
        
        print('Trainging Loss', loss_train/len(train_dataloader))
        
        if epoch % opt.save_epoch_freq == 0:
            torch.save(model.state_dict(), './results/'+ opt.name+ '/weights/'+str(epoch)+'.pt')

        # Validation
        model.eval()
        val_loss = 0.0
        ssim_total = 0.0
        psnr_total = 0.0
        with torch.no_grad():
            for index,val_data in enumerate(tqdm.tqdm(val_dataloader)):
                noise_val, clear_val = val_data

                noise_val, clear_val = noise_val.float().to(device), clear_val.float().to(device)
                
                noise_w_val, noise_b_val, clean_val = model(noise_val)
                
                loss_clear_val = mse(clean_val, noise_val)

                ssim_total += ssim(clean_val, noise_val)
                psnr_total += psnr(clean_val, noise_val)
                val_loss += loss_clear_val.item()

        print('Epoch: {}, Validation Loss: {}'.format(epoch, val_loss / len(val_dataloader)))
        print('Epoch: {}, SSIM Value : {}'.format(epoch, ssim_total / len(val_dataloader)))
        print('Epoch: {}, PSNR Value : {}'.format(epoch, psnr_total / len(val_dataloader)))
        
  

def test(opt,model,device,test_dataloader):
    
    model.eval()
    val_loss = 0.0
    ssim_total = 0.0
    psnr_total = 0.0
    with torch.no_grad():
        for index,val_data in enumerate(tqdm.tqdm(test_dataloader)):
            noise_val, clear_val = val_data

            noise_val, clear_val = noise_val.float().to(device), clear_val.float().to(device)
            
            noise_w_val, noise_b_val, clean_val = model(noise_val)
            # clean_val = noise_val
            
            save_image(opt,index,noise_val,clean_val)

            loss_clear_val = mse(clean_val, noise_val)

            ssim_total += ssim(clean_val, noise_val)
            psnr_total += psnr(clean_val, noise_val)
            val_loss += loss_clear_val.item()

        print(' Test Loss: {}'.format( val_loss / len(test_dataloader)))
        print(' SSIM Value : {}'.format( ssim_total / len(test_dataloader)))
        print(' PSNR Value : {}'.format( psnr_total / len(test_dataloader)))
        

def trainer(opt):
    # save result path
    result_path = './results/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_path + opt.name + '/'):
        os.makedirs(result_path + opt.name + '/')
    if not os.path.exists(result_path + opt.name + '/weights/'):
        os.makedirs(result_path + opt.name + '/weights/')

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create cvf model
    model = CVF_model(opt)
    if opt.Continue:
        model.load_state_dict(torch.load('./results/' + opt.name + '/weights/' + opt.model_name + '.pt'))
    
    model = model.to(device)
    # create train and  val dataloader
    train_dataloader,val_dataloader = dataset.train_dataloader(opt)
    # start train 
    print('start train')

    train(opt,model,device,train_dataloader,val_dataloader)
    
   


def tester(opt):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create cvf model
    model = CVF_model(opt)
    # load model 
    model.load_state_dict(torch.load('./results/' + opt.name + '/weights/' + opt.model_name + '.pt'))
    
    model = model.to(device)
    # create test dataloader
    test_dataloader = dataset.test_dataloader(opt)

    test(opt,model,device,test_dataloader)
    
