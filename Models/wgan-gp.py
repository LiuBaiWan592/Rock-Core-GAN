import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import numpy as np

class Generator(nn.Module):
    """
    WGAN-GP的生成器网络
    输入：随机噪声
    输出：生成的岩心图像
    """
    def __init__(self, latent_dim=100, img_shape=(3, 128, 128)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # 计算生成器输出形状
        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    WGAN-GP的判别器网络
    输入：真实或生成的岩心图像
    输出：图像真实性评分
    """
    def __init__(self, img_shape=(3, 128, 128)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # 输出层
        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1)
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class WGAN_GP:
    """
    WGAN-GP模型
    包含生成器和判别器的训练和推理逻辑
    """
    def __init__(self, 
                 img_shape=(3, 128, 128), 
                 latent_dim=100, 
                 lambda_gp=10,
                 lr=0.0002, 
                 b1=0.5, 
                 b2=0.999,
                 n_critic=5,
                 device='cuda'):
        
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.device = device
        
        # 初始化生成器和判别器
        self.generator = Generator(latent_dim, img_shape).to(device)
        self.discriminator = Discriminator(img_shape).to(device)
        
        # 优化器
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """计算梯度惩罚项"""
        # 随机采样插值系数
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        # 创建插值样本
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # 判别器对插值样本的输出
        d_interpolates = self.discriminator(interpolates)
        # 获取虚拟的全1张量作为梯度计算目标
        fake = torch.ones(real_samples.size(0), 1, device=self.device, requires_grad=False)
        # 计算梯度
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # 展平梯度
        gradients = gradients.view(gradients.size(0), -1)
        # 计算梯度范数
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def train_step(self, real_imgs):
        """单步训练"""
        # 配置输入
        real_imgs = real_imgs.to(self.device)
        batch_size = real_imgs.size(0)
        
        # ---------------------
        #  训练判别器
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # 从噪声生成一批图像
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_imgs = self.generator(z).detach()
        
        # 计算真实样本和生成样本的评分
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs)
        
        # 计算梯度惩罚
        gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
        
        # Wasserstein距离
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # 只有在判别器训练n_critic步后才训练生成器
        g_loss = None
        if batch_size % self.n_critic == 0:
            # ---------------------
            #  训练生成器
            # ---------------------
            self.optimizer_G.zero_grad()
            
            # 生成一批新的图像
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            gen_imgs = self.generator(z)
            
            # 判别生成图像
            gen_validity = self.discriminator(gen_imgs)
            
            # 生成器损失：最小化判别器对生成图像的负评分
            g_loss = -torch.mean(gen_validity)
            
            g_loss.backward()
            self.optimizer_G.step()
            
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item() if g_loss is not None else None,
            'wasserstein_distance': torch.mean(real_validity).item() - torch.mean(fake_validity).item(),
            'gp': gradient_penalty.item()
        }
        
    def generate(self, n_samples):
        """生成n_samples个岩心图像样本"""
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            return self.generator(z)
            
    def save_models(self, path):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }, path)
        
    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D']) 