import torch
import torch.nn as nn
import utils
import torchvision
import os


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion


        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None, sid = None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print('-restore-',x.shape,y)
                if sid:
                    y = y[0]
                    if sid+'__' in y:
                        print(self.args.image_folder, self.config.data.dataset)
                        print(i, y)
                        datasetname =  y.split('__')[0]
                        id = y.split('__')[1]
                        frame = y.split('__')[2]
                        print(datasetname, id, frame)
                        # print(os.path.join(image_folder, id, f"{frame}_output.png"))
                        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x                                                           ##1
                        x_cond = x[:, :3, :, :].to(self.diffusion.device)                                                                     ##2
                        x_gt   = x[:, 3:, :, :].to(self.diffusion.device)   
                        print('x_cond = ',x_cond.shape)

                        x_output = self.diffusive_restoration(x_cond, r=r)                                                                     ##3
                        x_output = inverse_data_transform(x_output)                                                                            ##4
                        utils.logging.save_image(x_cond, os.path.join('results/',validation, datasetname,'input',sid, f"{frame}.png"))
                        utils.logging.save_image(x_output, os.path.join('results/',validation,datasetname, 'output', sid, f"{frame}.png"))
                        utils.logging.save_image(x_gt, os.path.join('results/',validation,datasetname, 'gt',sid, f"{frame}.png"))             ##yy
                        x_output = x_output.to(self.diffusion.device)                                                                         ##yy
                        x_gt     = x_gt.to(self.diffusion.device)                                                                             ##yy
                        saveimg  = torch.cat((x_cond, x_output, x_gt), dim=3)                                                                 ##yy
                        utils.logging.save_image(saveimg, os.path.join('results/',validation,datasetname,'in_out_gt',sid, f"{frame}_3.png"))  ##yy
                else:
                    print(i, y)
                    y = y[0]
                    frame = y
                    print(f"starting processing from image {y}")
                    
                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x                                                                 ##1
                    x_cond = x[:, :3, :, :].to(self.diffusion.device)                                                                           ##2
                    x_gt   = x[:, 3:, :, :].to(self.diffusion.device)
                    #print('x_cond, x_gt = ',x_cond.shape, x_gt.shape) #[1, 3, 256, 256]
                    x_output = self.diffusive_restoration(x_cond, r=r)                                                                          ##3
                    x_output = inverse_data_transform(x_output)                                                                                 ##4
                    utils.logging.save_image(x_cond, os.path.join(image_folder, 'input', f"{frame}.png"))
                    utils.logging.save_image(x_output, os.path.join(image_folder, 'output', f"{frame}.png"))
                    utils.logging.save_image(x_gt, os.path.join(image_folder, 'gt', f"{frame}.png"))
                    x_output = x_output.to(self.diffusion.device)
                    x_gt     = x_gt.to(self.diffusion.device)
                    saveimg  = torch.cat((x_cond, x_output, x_gt), dim=3)
                    utils.logging.save_image(saveimg, os.path.join(image_folder, 'in_out_gt', f"{frame}.png"))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
