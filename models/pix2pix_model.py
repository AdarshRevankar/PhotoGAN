import torch
import models.networks as networks
import util.util as util
import numpy as np
import time

class Pix2PixModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        # Get Parameters
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        # Create Components
        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # Set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator(input_semantics, real_image)
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("Model is invalid")
    
    def create_optimizers(self, opt):
        # Get the Generater param
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            # Use existing param if present
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            # Use discriminator param if present
            D_params = list(self.netD.parameters())
        
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    # ------------------------------------------
    # Private Helper Methods
    # ------------------------------------------
    def initialize_networks(self, opt):
        # Loads the models from the stored place / creates new

        netG = networks.define_G(opt)
        # During training phase use Descriminator
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        # If during test / re-train time
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        
        return netG, netD, netE

        