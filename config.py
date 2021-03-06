import argparse

def config():
    parser = argparse.ArgumentParser()
    image_size = 256

    #training settings
    parser.add_argument('--dataset', type = str,default='CelebA',choices=['CelebA'], help='')
    parser.add_argument('--num_epochs',type=int,default=50, help='')
    parser.add_argument('--batch_size', default=8, type=int, help='')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--cuda',type=bool,default=True,help='')



    #hyper parameter
    parser.add_argument('--g_lr',type=float, default=2e-4,help='')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='')
    parser.add_argument('--l1_loss_alpha',type=int,default=1,help='')

    parser.add_argument('--gan_with_mask',type=bool,default=True,help='')
    parser.add_argument('--gan',type=str,default='sngan',help='')

    #path
    parser.add_argument('--image_path', type=str, default='/home/weebum/CelebA/img_align_celeba_png')
    parser.add_argument('--edge_path', type=str, default='/home/weebum/CelebA/img_align_celeba_png_edge')

    #parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--pretrained_model_G',type=str,default='',help='name of pretrained model in the model save path')
    parser.add_argument('--pretrained_model_D', type=str, default='', help='name of pretrained model in the model save path')

    parser.add_argument('--img_size', default=image_size,type=int, help='size of image')
    parser.add_argument('--crop_size', default=image_size, type=int, help='size of image')
    parser.add_argument('--img_shape',default=(image_size,image_size,3),type=int,help='size of image')
    parser.add_argument('--height',default=64,type=int,help='height of random bbox')
    parser.add_argument('--width',default=64,type=int,help='width of random bbox')
    parser.add_argument('--max_delta_height',default=8,type=int,help='')
    parser.add_argument('--max_delta_width',default=8,type=int,help='')
    parser.add_argument('--vertical_margin',default=0,type=int,help='')
    parser.add_argument('--horizontal_margin',default=0,type=int,help='')

    #to tune
    parser.add_argument('--guided',default=False,type=bool,help='')
    parser.add_argument('--edge_threshold',default='0.6',type=float,help='')

    # additional arguments
    parser.add_argument('--severity',default=1,type=int,help='masking severity (1~4)')
    parser.add_argument('--RAdam',action='store_true',help='whether to use RAdam optimizer')
    parser.add_argument('--save_fig',action='store_true',help='save image samples with varing masking severity.')
    parser.add_argument('--gamma',default=1.0,type=float,help='adversarial loss coefficient.')


    args = parser.parse_args()
    return args
