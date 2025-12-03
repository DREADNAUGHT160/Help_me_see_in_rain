from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)
        
    def log_images(self, tag, images, step, nrow=8):
        # images should be (N, C, H, W)
        # Denormalize if needed (assuming standard normalization)
        # For visualization, we might want to un-normalize
        # But for now, let's just log as is or clamp
        grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
        self.writer.add_image(tag, grid, step)
        
    def close(self):
        self.writer.close()
