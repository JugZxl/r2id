import os

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_r2id import do_inference
from utils.logger import setup_logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP-ReID R2ID Inference')
    parser.add_argument('--config_file', default='configs/person/vit_clipreid.yml', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('transreid', output_dir, if_train=False)
    logger.info(args)
    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info('\n' + cf.read())
    logger.info('Running with config:\n{}'.format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    do_inference(cfg, model, val_loader, num_query)
