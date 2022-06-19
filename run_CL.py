import torch
from dataloader import get_loader
from inpaint_model import *
import time
from utils import *
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from config import config
import torch_optimizer as optim2
from torchvision.utils import save_image


class Run(object):
    def __init__(self,args):
        self.args = args

        self.data_loader = get_loader(batch_size=self.args.batch_size,FLAGS=self.args,dataset='CelebA',mode=self.args.mode)
        self.test_loader = get_loader(batch_size=self.args.batch_size,FLAGS=self.args,dataset='CelebA',mode='test')


        self.init_network()


    def init_network(self):
        #Models
        if self.args.pretrained_model_G:
            self.G = torch.load(os.path.join(self.args.model_save_path,self.args.pretrained_model_G))
            self.D = torch.load(os.path.join(self.args.model_save_path,self.args.pretrained_model_D))

        else:
            self.G = CAGenerator()
            self.D = SNDiscriminator()

        if self.args.cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        #optimizer
        if (args.RAdam):
            self.g_optimizer = optim2.RAdam(self.G.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            self.d_optimizer = optim2.RAdam(self.D.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        else:
            self.g_optimizer = torch.optim.Adam(self.G.parameters(),self.args.g_lr)
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),self.args.d_lr)
        #loss


    def train(self):
        iters_per_epoch = len(self.data_loader)

        g_lr = self.args.g_lr
        d_lr = self.args.d_lr

        if self.args.pretrained_model_G:
            start = int(self.args.pretrained_model_G.split('_')[0]) + 1
        else:
            start = 1

        # bbox config w.r.t. severity
        original_height = args.height
        original_width = args.width

        log_severity = int(math.log(args.severity + 1, 2))
        args.height = args.height * log_severity
        args.width = args.width * log_severity

        start_time = time.time()
        args.num_epochs = 40

        print("**************************Curriculum Learning*************************")
        for i in range(1, 5):
            args.severity = i
            log_severity = int(math.log(args.severity + 1, 2))
            args.height = original_height * log_severity
            args.width = original_width * log_severity
            print("<Severity %d>-----------------------------------------------------------" % (args.severity))

            for epoch in range(10):
                self.G.train()
                self.D.train()
                print("[Epoch %d]" % (epoch))
                counter = 0 # debug
                avg_g_loss = 0.0
                avg_d_loss = 0.0
                #with torch.autograd.detect_anomaly():
                for real_images in tqdm(self.data_loader): #real_image: B x 3 x H x W
                    edge = torch.zeros_like(real_images)[:,0:1,:,:].type(torch.float32)

                    batch_size, height, width = real_images.shape[0], real_images.shape[2], real_images.shape[3]
                    real_images = 2.*real_images - 1. #[-1,1]

                    #generate mask ,1 represents masked point
                    bbox = random_bbox(self.args)
                    regular_mask = bbox2mask(self.args,bbox)
                    irregular_mask = brush_stroke_mask(self.args, severity=args.severity)

                    binary_mask = np.logical_or(regular_mask.astype(bool),irregular_mask.astype(bool)).astype(np.float32)
                    binary_mask = torch.FloatTensor(binary_mask)

                    batch_mask = binary_mask.repeat(batch_size,1,1,1)

                    masked_edge = batch_mask * edge

                    inverse_mask = 1. - batch_mask
                    masked_images = real_images.clone() * inverse_mask

                    if (args.save_fig and counter < 10):
                        save_image(masked_images, "sample_%d_s%d.png" % (counter, args.severity), nrow=8, normalize=True)
                        counter += 1
                    
                    data_input = torch.cat((masked_images,batch_mask,masked_edge),dim=1)

                    self.D.zero_grad(), self.G.zero_grad(), self.d_optimizer.zero_grad(), self.g_optimizer.zero_grad()
                    
                    if self.args.cuda:
                        data_input = data_input.cuda()
                        batch_mask = batch_mask.cuda()
                        masked_edge = masked_edge.cuda()
                        binary_mask = binary_mask.cuda()
                        inverse_mask = inverse_mask.cuda()
                        masked_images = masked_images.cuda()
                        real_images = real_images.cuda()

                    stage_1,stage_2,offset_flow = self.G(data_input,binary_mask)


                    batch_complete = stage_2 * batch_mask + masked_images * inverse_mask

                    ae_loss = self.args.l1_loss_alpha * reduce_mean(torch.abs(real_images-stage_1),dim=[0,1,2,3]).view(-1)
                    ae_loss +=self.args.l1_loss_alpha * reduce_mean(torch.abs(real_images-stage_2),dim=[0,1,2,3]).view(-1)

                    batch_pos_neg = torch.cat((real_images,batch_complete),dim=0)
                    if self.args.gan_with_mask:
                        batch_pos_neg = torch.cat((batch_pos_neg,batch_mask.repeat(2,1,1,1)),dim=1)

                    if self.args.guided:
                        # conditional gan
                        batch_pos_neg = torch.cat((batch_pos_neg,masked_edge.repeat(2,1,1,1)),dim=1)
                    else:
                        batch_pos_neg = torch.cat((batch_pos_neg, torch.zeros_like(masked_edge).repeat(2, 1, 1, 1)), dim=1)

                    
                    batch_pos_neg_dup = batch_pos_neg.clone().detach()

                    """
                    # Test only Generator Training (Debug)
                    g_loss = ae_loss
                    g_loss.backward()
                    self.g_optimizer.step()
                    """

                    pos_neg = self.D(batch_pos_neg)
                    pos,neg = torch.split(pos_neg,batch_size,dim=0)
                    g_loss = ae_loss - args.gamma * reduce_mean(neg,dim=[0,1])
                    g_loss.backward()#g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()
                    
                    self.D.zero_grad(), self.G.zero_grad(), self.d_optimizer.zero_grad(), self.g_optimizer.zero_grad()
                    pos_neg = self.D(batch_pos_neg_dup)
                    pos,neg = torch.split(pos_neg,batch_size,dim=0)
                    hinge_pos = reduce_mean(F.relu(1-pos),dim=[0,1]).view(-1)
                    hinge_neg = reduce_mean(F.relu(1+neg),dim=[0,1]).view(-1)
                    d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg
                    
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    avg_g_loss += g_loss.item()
                    avg_d_loss += d_loss.item()



                    #print(str(((g_loss.item()), d_loss.item())))

                # save model checkpoint
                if not os.path.exists(self.args.model_save_path):
                    os.mkdir(self.args.model_save_path)
                torch.save(self.G,os.path.join(self.args.model_save_path,'{}_G_L1_{}.pth'.format(epoch,self.args.l1_loss_alpha)))
                torch.save(self.D,os.path.join(self.args.model_save_path, '{}_D_L1_{}.pth'.format(epoch, self.args.l1_loss_alpha)))
                avg_g_loss /= iters_per_epoch
                avg_d_loss /= iters_per_epoch
                print('Train Loss (G/D): %.8f / %.7f' % (avg_g_loss, avg_d_loss))



    def backprop(self,G=True,D=True):
        if D:
            self.d_optimizer.zero_grad()
            self.d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        if G:
            self.g_optimizer.zero_grad()
            self.g_loss.backward()
            self.g_optimizer.step()


    def gan_hinge_loss(self,pos,neg,name='gan_hinge_loss'):
        #print('pos_shape:',pos.shape)
        #print('neg_shape:',neg.shape)
        hinge_pos = reduce_mean(F.relu(1-pos),dim=[0,1]).view(-1)
        hinge_neg = reduce_mean(F.relu(1+neg),dim=[0,1]).view(-1)
        d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg
        g_loss = -reduce_mean(neg,dim=[0,1])
        return g_loss,d_loss





if __name__ == '__main__':
    args = config()
    print(args.model_save_path)
    runer = Run(args)
    if args.mode == 'train':
        runer.train()
