import subprocess
import os
import subprocess
import zipfile
import cairosvg
import pandas as pd
import json
import cv2
from PIL import Image
import argparse
import random

# pip install requirements, etc
#subprocess.call(['git', 'clone', 'https://github.com/devonbrackbill/stable-diffusion.git'])
# subprocess.call(['cd', 'stable-diffusion'])
# subprocess.call(['pip', 'install', 'requirements.txt'])
# subprocess.call(['pip', 'install', '--upgrade', 'keras'])
# subprocess.call(['pip', 'uninstall', '-y', 'torchtext'])
# subprocess.call(['pip', 'install', 'huggingface_hub'])
# subprocess.call(['pip', 'install', 'cairosvg', 'pandas'])

# imports


# add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2,
                    help='number of samples to generate')
args = parser.parse_args()

EXPERIMENT = 'experiment6'

print('generating', args.n_samples, 'samples for Experiment: {}'.
      format(EXPERIMENT))

# download FA images and parse to get the list of icon labels
# (we will use these for evaluation the models)
try:
    os.mkdir('res')
    print('created /res directory')
except OSError:
    print("res/ already exists")

# fetch Fontawesome
if len([file for file in os.listdir('res/') if 'fontawesome' in file]) == 0:
    subprocess.call(
        ["wget",
         "https://use.fontawesome.com/releases/v6.2.0/fontawesome-free-6.2.0-web.zip", 
         "--directory-prefix=res"])
else:
    print("fontawesome is already downloaded: {}".format([file for file in os.listdir('res/') if 'fontawesome' in file]))


# unzip
fontawesome_dir = ['res/' + file for file in os.listdir('res/') if 'fontawesome' in file and '.zip' in file][0]

with zipfile.ZipFile(fontawesome_dir,"r") as zip_ref:
    zip_ref.extractall("res")

fontawesome_dir = ['res/' + file for file in os.listdir('res/') if 'fontawesome' in file and 'zip' not in file][0]

print('FontAwesome files saved in: {}'.format(fontawesome_dir))

# read svg file -> png data
WIDTH = 512
HEIGHT = 512
png_dir = 'res/fontawesome-png'

try:
    os.mkdir(png_dir)
except OSError:
    print("{} already exists".format(png_dir))

files = sorted(os.listdir(os.path.join(fontawesome_dir, 'svgs', 'solid')))

filenames = [file.split('.svg')[0] for file in files]
textdescrip = [file.replace('-', ' ') for file in filenames]
filenames_png = [file + '.png' for file in filenames]

print('found {} files'.format(len(files)))
print('Converting SVG to PNG...')

for file, filename in zip(files, filenames):
    cairosvg.svg2png(url=os.path.join(fontawesome_dir, 'svgs', 'solid', file), 
                    output_width=WIDTH, 
                    output_height=HEIGHT,
                    write_to=os.path.join(png_dir, filename + '.png'))
    # read in the image
    img = cv2.imread(os.path.join(png_dir, filename + '.png'), cv2.IMREAD_UNCHANGED)

    if len(img.shape) and img.shape[2] == 4:

        # change black -> white and white -> black
        img[:, :, 0] = 255-img[:, :, 3]
        img[:, :, 1] = 255-img[:, :, 3]
        img[:, :, 2] = 255-img[:, :, 3]
        # remove 4th color channel (4th is alpha channel)
        img = img[:, :, :3]

        cv2.imwrite(os.path.join(png_dir, filename + '.png'), img)
    else:
        print("Image does not have 4 channels; deleting it: {}".format(os.path.join(png_dir, filename + '.png')))
        os.unlink(os.path.join(png_dir, filename + '.png'))

filenames_df = pd.DataFrame({#'file_name' : files,
                             'file_name': filenames_png,
                             'text': textdescrip})
filenames_df.tail()

# create model configs

yaml = \
"""model:
  base_learning_rate: 1.0e-04  # increased learning rate from 1.0e-04
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000  # increased from 1000
    first_stage_key: "image"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: crossattn
    scale_factor: 0.18215

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 1 ] # NOTE for resuming. use 10000 if starting from scratch
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      ckpt_path: "models/first_stage_models/kl-f8/model.ckpt"
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder


data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 4
    num_workers: 4
    num_val_workers: 0 # Avoid a weird val dataloader issue
    train:
      target: ldm.data.simple.hf_dataset_from_local
      params:
        name: imagefolder
        data_dir: res/fontawesome-png
        image_transforms:
        - target: torchvision.transforms.Resize
          params:
            size: 512
            interpolation: 3
        - target: torchvision.transforms.RandomCrop
          params:
            size: 512
        # - target: torchvision.transforms.RandomHorizontalFlip
        text_column: txt
        image_column: image
        caption_key: txt
    validation:
      target: ldm.data.simple.TextOnly
      params:
        captions:
        - "{prefix}radar{postfix}"
        - "{prefix}bunny rabbit{postfix}"
        - "{prefix}Yoda{postfix}"
        - "{prefix}coffee{postfix}"
        - "{prefix}palm tree{postfix}"
        - "{prefix}evergreen tree{postfix}"
        - "{prefix}toucan{postfix}"
        - "{prefix}thumbs up{postfix}"
        - "{prefix}butterfly{postfix}"
        output_size: 512
        n_gpus: 1 # small hack to sure we see all our samples; # I changed this to 1


lightning:
  find_unused_parameters: False

  modelcheckpoint:
    params:
      every_n_train_steps: 500
      save_top_k: -1
      monitor: null

  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 2000
        max_images: 10
        increase_log_steps: False
        log_first_step: True
        log_all_val: True
        log_images_kwargs:
          use_ema_scope: True
          inpaint: False
          plot_progressive_rows: False
          plot_diffusion_rows: False
          N: 4
          unconditional_guidance_scale: 3.0
          unconditional_guidance_label: [""]

  trainer:
    benchmark: True
    num_sanity_val_steps: 0
    accumulate_grad_batches: 1
""".format(**{'prefix': 'black and white SVG icon of ',
              'postfix': ', fontawesome, nounproject'})

with open('configs/stable-diffusion/retrain-icons.yaml', 'w') as f:
    f.write(yaml)

# setup the directories, one for each model
models = {
    # 'model0_evalspecificquery':
    #     {'model_configs': '',
    #      's3_location': None,
    #      'querytype': 'specific'},
    'model1_evalspecificquery_666':
        {'model_configs': '',
         's3_location': 's3://379552636459-stable-diffusion/2022-10-26T04-04-20_retrain-icons/checkpoints/epoch=000666.ckpt',
         'querytype': 'specific'},
    'model1_evalspecificquery_last':
        {'model_configs': '',
         's3_location': 's3://379552636459-stable-diffusion/2022-10-26T04-04-20_retrain-icons/checkpoints/last.ckpt',
         'querytype': 'specific'},
    'model1_evalgeneralquery_last':
        {'model_configs': '',
         's3_location': 's3://379552636459-stable-diffusion/2022-10-26T04-04-20_retrain-icons/checkpoints/last.ckpt',
         'querytype': 'general'},
    # 'model3_evalgeneralquery_666': {
    #     'model_configs': '',
    #     's3_location': 's3://379552636459-stable-diffusion/2022-10-26T15-45-26_retrain-icons/checkpoints/epoch=000666.ckpt',
    #     'querytype': 'general'},
    'model3_evalgeneralquery_last':
        {'model_configs': '',
         's3_location': 's3://379552636459-stable-diffusion/2022-10-26T15-45-26_retrain-icons/checkpoints/last.ckpt',
         'querytype': 'general'},
}

# download the model checkpoints from S3
for model in sorted(models.keys()):
    try:
        os.mkdir(model)
        print('created /{} directory'.format(model))
    except OSError:
        print("{} already exists".format(model))

    try:
        os.makedirs('outputs/' + model)
        print('created /outputs/{} directory'.format(model))
    except OSError:
        print("outputs/{} already exists".format(model))
        pass

    # if models[model]['s3_location'] is not None and '.cptk' not in os.listdir(model):
    # check if the checkpoint is already downloaded
    # any filename has .ckpt in it
    if models[model]['s3_location'] is not None and not any('.ckpt' in s for s in os.listdir(model)):
        subprocess.call(['aws', 's3', 'cp', models[model]['s3_location'], model])
        print('downloaded {} to {}'.format(models[model]['s3_location'], model))
    else:
        print('no s3_location for {} or {} already exists'.format(model, model))
        pass

# create a prompt list
random.seed(43)
prompts_master_list = filenames_df['text'].tolist()
prompts_master_list = random.sample(prompts_master_list, 100)
prompts_master_list = prompts_master_list + \
    ['darth vader', 'yoda', 'coffee', 'bunny rabbit', 'radar', 'palm tree',
     'evergreen tree', 'toucan', 'thumbs up', 'butterfly']
# prompts_master_list = ['darth vader', 'rabbit']

# base_seeds = [1, 2, 3, 4, 5]

seed_start = 43

# loop through the list of models
for model in sorted(models.keys()):
    print('model: {}'.format(model))

    # create the prompts
    if models[model]['querytype'] == 'specific':
        # specific query
        prefix = 'black and white SVG icon of '
        postfix = ', fontawesome, nounproject'
    elif models[model]['querytype'] == 'general':
        prefix = ''
        postfix = ''
    else:
        print('no querytype for {}'.format(model))
        pass

    prompts = [[prefix + "{}".format(prompt) + postfix] * args.n_samples for
               prompt in prompts_master_list]
    # flatten the list
    prompts = [item for sublist in prompts for item in sublist]
    # write prompts to file
    with open('prompts.txt', 'w') as f:
        for prompt in prompts:
            f.write("%s\n" % prompt)

    # keep increasing the seed as we move thru the prompts
    for seed in [seed_start]:

        subprocess.call([
            "python", "scripts/txt2img.py",
            # '--prompt', '{}'.format(prompt),
            '--outdir', 'outputs/{}'.format(model),
            "--H", "512",  "--W", "512",
            "--seed", "{}".format(seed),
            "--n_samples", "{}".format(args.n_samples),
            '--config', 'configs/stable-diffusion/retrain-icons.yaml',
            '--skip_grid',
            '--from-file', 'prompts.txt',
            '--ckpt', '{}/{}'.format(
                model,
                models[model]['s3_location'].split('/')[5])])
        print('generated images for {}'.format(model))
        # seed_start += 1

    # copy the images to S3
    subprocess.call(
        ['aws', 's3', 'cp', 'outputs/{}'.format(model),
         's3://379552636459-stable-diffusion/{}/outputs/{}'.
            format(EXPERIMENT, model),
         '--recursive'])
    subprocess.call(
        ['aws', 's3', 'cp', 'prompts.txt',
         's3://379552636459-stable-diffusion/{}/outputs/{}/prompts.txt'.
            format(EXPERIMENT, model),
         '--recursive'])
    print('copied images to S3 for {}'.format(model))
