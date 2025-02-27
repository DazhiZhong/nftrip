import utils.utils as utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
from pathlib import Path
import shutil
import numpy as np
import os
import argparse
from time import time
import logging
from datetime import datetime
import imageio
from PIL import Image
import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from rudalle.realesrgan.model import RealESRGAN
from huggingface_hub import hf_hub_url, cached_download

logs_folder = Path('logs')
logs_folder.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f'logs/log-{datetime.now().strftime("%d_%m_%Y")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


MODELS = {
    'x2': dict(
        scale=2,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x2.pth',
    ),
    'x4': dict(
        scale=4,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x4.pth',
    ),
    'x8': dict(
        scale=8,
        repo_id='shonenkov/rudalle-utils',
        filename='RealESRGAN_x8.pth',
    ),
}

for model in MODELS.keys():
    cache_dir = 'models'
    config = MODELS[model]
    config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=config['filename'])


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config):
    logger.info('Starting training..')

    content_img_path = config['content_images_dir'] / config['content_img_name']
    style_img_path = config['style_images_dir'] / config['style_img_name']

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = config['output_img_dir'] / out_dir_name
    dump_path.mkdir(exist_ok=True, parents=True)
    [f.unlink() for f in dump_path.glob('*') if f.is_file()]  # remove all files in the folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'Using device {device} for training')

    content_img = utils.prepare_img(str(content_img_path), config['height'], device)
    style_img = utils.prepare_img(str(style_img_path), config['height'], device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    logger.debug(f'Using {config["model"]} in the optimization procedure')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 1000,
    }

    #
    # Start of optimization procedure
    #
    unsuccessful=False
    if config['optimizer'] == 'adam':
        logger.info('Using ADAM')
        try:
            optimizer = Adam((optimizing_img,), lr=1e1)
            tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            for cnt in range(num_of_iterations[config['optimizer']]):
                total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
                with torch.no_grad():
                    #logger.debug(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                    utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
        except:
            unsuccessful = True
    elif config['optimizer'] == 'lbfgs':
        logger.info('Using LBFGS')
        try:
            # line_search_fn does not seem to have significant impact on result
            optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
            cnt = 0

            def closure():
                nonlocal cnt
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
                if total_loss.requires_grad:
                    total_loss.backward()
                with torch.no_grad():
                    #logger.debug(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                    utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

                cnt += 1
                return total_loss

            optimizer.step(closure)
        except:
            unsuccessful = True
    
    logger.debug(f'Executed {cnt} iterations')
    if ((cnt<num_of_iterations['lbfgs']/10) or unsuccessful==True) & (config['optimizer'] == 'lbfgs'):
        config['optimizer']='adam'
        return neural_style_transfer(config)

    logger.info('Training finished')
    return dump_path


def make_gif(config, results_path):
    logger.info('Creating gif..')

    results = list(sorted(results_path.glob('*.png')))[-1]
    transf = Image.open(results)
    
    content_img_path = config['content_images_dir'] / config['content_img_name']
    orig = Image.open(content_img_path)
    orig = orig.resize((transf.width,transf.height), Image.ANTIALIAS)

    images = []
    for i in range(0, 255, 15):
        orig_new = orig.copy()
        orig_new.putalpha(255-i)

        transf_new = transf.copy()
        transf_new.putalpha(i)

        new = Image.alpha_composite(orig_new, transf_new)
        images.append(new)

    images = images + images[::-1]
    imageio.mimsave(f'{results_path}/out.gif', images, duration=0.01)


def copy_output(optimization_config, results_path):
    optimization_config['images_path'].mkdir(exist_ok=True, parents=True)
    optimization_config['gifs_path'].mkdir(exist_ok=True, parents=True)
    destination_filename = optimization_config['output_img_name'].split('.')[0]

    source_img_filename = sorted(list(results_path.glob('*.png')))[-1]
    logger.info(f'Copying {source_img_filename}')
    destination_img_file = optimization_config['images_path'] / (destination_filename + '.png')
    shutil.copy(source_img_filename, destination_img_file)
    
    if optimization_config['gif']:
        source_gif_filename = list(results_path.glob('*.gif'))[0]
        logger.info(f'Copying {source_gif_filename}')
        destination_video_file = optimization_config['gifs_path'] / ('g' + destination_filename + '.gif')
        logger.info(f'Copying to {destination_video_file}')
        shutil.copy(source_gif_filename, destination_video_file)

    shutil.rmtree(results_path, ignore_errors=True)


    
def main(optimization_config):
    start = time()

    default_resource_dir = Path(os.path.dirname(__file__)) / 'data'
    content_images_dir = default_resource_dir / 'content-images'
    style_images_dir = default_resource_dir / 'style-images'
    output_img_dir = default_resource_dir / 'output-images'
    img_format = (4, '.png')  # saves images in the format: %04d.png
    
    output_path = default_resource_dir / optimization_config['output_path']
    images_path = output_path / 'images'
    gif_path = output_path / 'gifs'

    # just wrapping settings into a dictionary
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['images_path'] = images_path
    optimization_config['gifs_path'] = gif_path
    optimization_config['img_format'] = img_format

    logger.debug(optimization_config)

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    images_path = neural_style_transfer(optimization_config)
    
    if optimization_config['gif']:
        make_gif(optimization_config, images_path)
    
    
    scale = optimization_config["esrgan_scale"] 
    if scale in [2,4,8]:
        logger.info(f'esr')
        result_img_path = list(sorted(images_path.glob('*.png')))[-1]
        esrgan = ESRGANUpscale(optimization_config["esrgan_scale"])
        esrgan.gan_upscale(str(result_img_path),str(result_img_path)[:-4]+"_esr.png")

    # copy results to the respective folder
    copy_output(optimization_config, images_path)

    logger.info(f'Time elapsed: {time()-start}')
    
    return True
  

class ESRGANUpscale():
    def __init__(self,esrganscale):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"using {device} for esrganupscale")
        self.model = RealESRGAN(device, esrganscale)
        self.model.load_weights(f"models/RealESRGAN_x{esrganscale}.pth")
        print("Model loaded!")
    def gan_upscale(self,imgpath,outpath,return_image=False):
        input_image = Image.open(str(imgpath))     
        input_image = input_image.convert('RGB')
        with torch.no_grad():
            sr_image = self.model.predict(np.array(input_image))
        sr_image.save(outpath)
        if return_image:
            return sr_image
        else:
            return None


if __name__ == "__main__":
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='adamo.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='birds.jpg')
    parser.add_argument("--output_img_name", type=str, help="output image name", default='test.png')
    parser.add_argument("--output_path", type=str, help='output path', default='output')
    parser.add_argument("--height", type=int, nargs='+', help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument('--gif', dest='gif', action='store_true')
    parser.add_argument('--no-gif', dest='gif', action='store_false')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    parser.add_argument('--esrgan_scale',type=int, help="2, 4, 8" ,dest='esrgan_scale')
    parser.set_defaults(gif=False, esrscale=1)
    args = parser.parse_args()
    
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
        
    main(optimization_config)
    