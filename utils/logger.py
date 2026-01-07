
#################### needs to be checked an possibly modified ####################

import os
import logging
import torch
from torchvision.utils import save_image

# Set up logging configuration
def setup_logger(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, 'train.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger


# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='./checkpoints', filename='checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))


# Save generated sample images from the generator
def save_samples(generator, epoch, fixed_noise, sample_dir='./samples', num_samples=5):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise)
        save_image(samples.data, os.path.join(sample_dir, f'epoch_{epoch}.png'), nrow=num_samples, normalize=True)
    generator.train()
