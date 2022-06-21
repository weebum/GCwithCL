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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchvision.transforms.functional import rgb_to_grayscale


class Run(object):
    def __init__(self,args):
        self.args = args

        self.data_loader = get_loader(batch_size=self.args.batch_size,FLAGS=self.args,dataset='CelebA',mode=self.args.mode)
        self.test_loader = get_loader(batch_size=self.args.batch_size,FLAGS=self.args,dataset='CelebA',mode='test')


        self.init_network()

        self.lpips = lpips.LPIPS(net='alex')


    def init_network(self):
        #Models
        if self.args.pretrained_model_G:
            print('Loading generator from %s' % (os.path.join(self.args.model_save_path,self.args.pretrained_model_G)))
            print('Loading discriminator from %s' % (os.path.join(self.args.model_save_path,self.args.pretrained_model_D)))

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

    def calculate_scores(self, original_data, reconstructed_data): # takes (real_images, stage_2) as argument
        original_data, reconstructed_data = original_data.cpu(), reconstructed_data.cpu()
        ssim_score, psnr_score, lpips_score = 0.0, 0.0, 0.0
        original_np, reconstructed_np = rgb_to_grayscale(original_data).numpy().squeeze(), rgb_to_grayscale(reconstructed_data).numpy().squeeze()

        batch_size = len(original_data)

        for i in range(batch_size):
            ssim_score += ssim(original_np[i], reconstructed_np[i])
            psnr_score += psnr(original_np[i], reconstructed_np[i])
            #lpips_score += self.lpips(original_data, reconstructed_data)

        ssim_score /= batch_size
        psnr_score /= batch_size
        #lpips_score = (lpips_score / batch_size).numpy()

        return (ssim_score, psnr_score, lpips_score)



    def test(self):
        iters_per_epoch = len(self.test_loader)

        # bbox config w.r.t. severity
        log_severity = int(math.log(args.severity + 1, 2))
        args.height = args.height * log_severity
        args.width = args.width * log_severity

        start_time = time.time()
        
        self.G.eval()
        self.D.eval()
        counter = 0
        avg_g_loss, avg_d_loss = 0.0, 0.0
        avg_ssim, avg_psnr, avg_lpips = 0.0, 0.0, 0.0
        original_img_samples = torch.zeros((5, 3, 256, 256))
        masked_img_samples = torch.zeros((5, 3, 256, 256))
        recon_img_samples = torch.zeros((5, 3, 256, 256))
        #with torch.autograd.detect_anomaly():
        with torch.no_grad():
            for real_images in tqdm(self.test_loader): #real_image: B x 3 x H x W
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

                data_input = torch.cat((masked_images,batch_mask,masked_edge),dim=1)

                if self.args.cuda:
                    data_input = data_input.cuda()
                    batch_mask = batch_mask.cuda()
                    masked_edge = masked_edge.cuda()
                    binary_mask = binary_mask.cuda()
                    inverse_mask = inverse_mask.cuda()
                    masked_images = masked_images.cuda()
                    real_images = real_images.cuda()

                stage_1, stage_2, offset_flow = self.G(data_input,binary_mask)

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

                """
                # Test only Generator Training (Debug)
                g_loss = ae_loss
                g_loss.backward()
                self.g_optimizer.step()
                """

                pos_neg = self.D(batch_pos_neg)
                pos,neg = torch.split(pos_neg,batch_size,dim=0)
                g_loss = ae_loss - args.gamma * reduce_mean(neg,dim=[0,1])

                hinge_pos = reduce_mean(F.relu(1-pos),dim=[0,1]).view(-1)
                hinge_neg = reduce_mean(F.relu(1+neg),dim=[0,1]).view(-1)
                d_loss = 0.5 * hinge_pos + 0.5 * hinge_neg


                avg_g_loss += g_loss.item()
                avg_d_loss += d_loss.item()

                current_ssim, current_psnr, current_lpips = self.calculate_scores(real_images, stage_2)
                avg_ssim += current_ssim
                avg_psnr += current_psnr
                avg_lpips += current_lpips

                if (counter < 5 and args.save_fig):
                    original_img_samples[counter] = real_images[0]
                    masked_img_samples[counter] = masked_images[0]
                    recon_img_samples[counter] = stage_2[0]

                    if (counter == 4):
                        samples = torch.cat((original_img_samples, masked_img_samples, recon_img_samples), -2)
                        model_name = args.model_save_path
                        save_image(samples, "sample_s%d_%s.png" % (args.severity, model_name), nrow=5, normalize=True)
                    counter += 1


            avg_g_loss /= iters_per_epoch
            avg_d_loss /= iters_per_epoch
            avg_ssim /= iters_per_epoch
            avg_psnr /= iters_per_epoch
            avg_lpips /= iters_per_epoch
            print('Test Loss (G/D): %.8f / %.7f' % (avg_g_loss, avg_d_loss))
            print('[Test Scores] SSIM: %.4f, PSNR: %.4f, LPIPS: %.4f' % (avg_ssim, avg_psnr, avg_lpips))



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
    runer = Run(args)
    if args.mode == 'train':
        runer.test()
