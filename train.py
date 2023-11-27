import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from datasets.lrs3_dataset import get_datasets_LRS3
from datasets.mead_dataset import get_datasets_MEAD
from datasets.mead_sides_dataset import get_datasets_MEAD_sides
#from datasets.bupt_dataset import get_datasets_BUPT
from datasets.ffhq_dataset import get_datasets_FFHQ
from datasets.celeba_dataset import get_datasets_CelebA

from models.model import Model
import os
import copy

def parse_args():
    # load config
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    # Remove the configuration file name from sys.argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # merge config with cli args
    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()
    
    if config.train.exp_name is None:
        config.train.exp_name = 'exp'


    # ----------------------- initialize log directory ----------------------- #
    log_path = os.path.join(config.train.log_path, config.train.exp_name)
    train_images_save_path = os.path.join(log_path, 'train_images')
    val_images_save_path = os.path.join(log_path, 'val_images')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(train_images_save_path, exist_ok=True)
    os.makedirs(val_images_save_path, exist_ok=True)

    # copy config file to log directory
    OmegaConf.save(config, os.path.join(log_path, 'config.yaml'))


    train_dataset_LRS3, val_dataset_LRS3, test_dataset_LRS3 = get_datasets_LRS3(config)
    train_dataset_MEAD, val_dataset_MEAD, test_dataset_MEAD = get_datasets_MEAD(config)
    train_dataset_MEAD_sides, val_dataset_MEAD_sides, test_dataset_MEAD_sides = get_datasets_MEAD_sides(config)
    # train_dataset_affectnet = get_datasets_AffectNet(config)
    #train_dataset_bupt = get_datasets_BUPT(config)
    train_dataset_ffhq = get_datasets_FFHQ(config)
    train_dataset_celeba = get_datasets_CelebA(config)
    
    dataset_percentages = {
        'LRS3': config.dataset.LRS3_percentage,
        'MEAD': config.dataset.MEAD_percentage,
        #'BUPT': 0.0,
        'FFHQ': config.dataset.FFHQ_percentage,
        'CELEBA': config.dataset.CelebA_percentage,
        'MEAD_sides': config.dataset.MEAD_sides_percentage
    }

    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_LRS3, 
                                                    train_dataset_MEAD, 
                                                    train_dataset_ffhq, 
                                                    train_dataset_celeba,
                                                    train_dataset_MEAD_sides, 
                                                    ])
    
    from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
    sampler = MixedDatasetBatchSampler([len(train_dataset_LRS3),
                                        len(train_dataset_MEAD), 
                                        len(train_dataset_ffhq), 
                                        len(train_dataset_celeba),
                                        len(train_dataset_MEAD_sides)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, config.train.samples_per_epoch)
    
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_LRS3, val_dataset_MEAD])
                                                    
    
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
    #                                         num_workers=config.train.num_workers, shuffle=False, drop_last=True, sampler=sampler)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.num_workers)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True)
        
        
    model = Model(config)

    
    model = model.cuda()

    #model.configure_optimizers(config.train.num_epochs)

    if config.train.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="ubermench",
            name = config.train.exp_name
        )
        model.setup_logger(wandb_run)

    if config.resume:
        loaded_state_dict = torch.load(config.resume)
        for key in list(loaded_state_dict.keys()):
            if ('discriminator' in key) or ('flame' in key) or ('renderer' in key):
                del loaded_state_dict[key]

        model.load_state_dict(loaded_state_dict, strict=False)
        
    if config.load_generator_only:
        loaded_state_dict = torch.load(config.load_generator_only)
        
        new_state_dict = {}
        for key in list(loaded_state_dict.keys()):
            if 'generator' in key:
                #new_state_dict[key.replace('generator.', '')] = loaded_state_dict[key]
                new_state_dict[key] = loaded_state_dict[key]
        
        model.load_state_dict(new_state_dict, strict=False)

            
    if config.load_encoder_only:
        loaded_state_dict = torch.load(config.load_encoder_only)
        #print(loaded_state_dict.keys())
        
        new_state_dict = {}
        for key in list(loaded_state_dict.keys()):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if key.startswith('encoder.'):# and 'pose' in key:
                new_state_dict[key] = loaded_state_dict[key]
        
        model.load_state_dict(loaded_state_dict, strict=False)
        # model.load_state_dict(new_state_dict, strict=False)


    # at the start of each epoch for alternating optimization
    model.base_encoder = copy.deepcopy(model.encoder)
    model.base_encoder.eval()

    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        
        # restart everything at each epoch!
        model.configure_optimizers(len(train_loader))

        # # ----------------------- expr + shape ----------------------- #
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            if batch is None:
                continue
            #batch = {k: v.cuda() for k, v in batch.items() if k != 'subject' and k != 'filename' and k != 'dataset_name'}
            outputs = model.step(batch, i, phase='train')

            if i % config.train.visualize_every == 0:
                save_path = os.path.join(train_images_save_path, f'{epoch}_{i}.jpg')
                model.visualize(outputs, epoch, i, train_images_save_path, show_landmarks=False)


        # ----------------------- val ----------------------- #

        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if batch is None:
                continue
            #batch = {k: v.cuda() for k, v in batch.items() if k != 'subject' and k != 'filename'}
            outputs = model.step(batch, i, phase='val')

            if i % config.train.visualize_every == 0:
                save_path = os.path.join(val_images_save_path, f'{epoch}_{i}.jpg')
                model.visualize(outputs, epoch, i, val_images_save_path, show_landmarks=False)

        state_dict = model.state_dict()
        for key in list(state_dict.keys()):
            if ('discriminator' in key) or ('flame' in key) or ('renderer' in key) or ('deca_encoder' in key) or ('vgg' in key):
                del state_dict[key]

        if epoch % config.train.save_every == 0:
            torch.save(state_dict, os.path.join(log_path, 'model_{}.pt'.format(epoch)))