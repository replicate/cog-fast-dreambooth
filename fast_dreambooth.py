

# pull model - either SD 1.5 or a custom model
# lots of custom logic to pull a few different kinds of models
# dreambooth - 
import mimetypes
from pathlib import Path
import shutil
import os
import time
import argparse
from zipfile import ZipFile

def download_sd_15():
    """Pulls SD model as base for dreambooth"""
    if os.path.exists('content/stable-diffusion-v1-5'):
        shutil.rmtree('content/stable-diffusion-v1-5')
    os.makedirs('content/stable-diffusion-v1-5', exist_ok=True)
    os.chdir('content/stable-diffusion-v1-5')
    os.system('git init')
    os.system('git remote add -f origin https://huggingface.co/runwayml/stable-diffusion-v1-5')
    os.system('git config core.sparsecheckout true')
    os.system('echo "scheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!vae/diffusion_pytorch_model.bin\n!*.safetensors" > .git/info/sparse-checkout')
    os.system('git pull origin main')
    if os.path.exists('unet/diffusion_pytorch_model.bin'):
        print('found')
        os.system('wget -q -O vae/diffusion_pytorch_model.bin https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin')
        os.system('rm -r .git')
        os.system('rm model_index.json')
        time.sleep(1)
        os.system('wget -q https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')


def run(class_images, instance_images, captions):
    if not os.path.exists('/src/content/stable-diffusion-v1-5'):
        download_sd_15()
    
    # set up instance images, class images, and captions
    # the only real difference b/w this and our dreambooth is space for custom captions
    # so let's say that we're storing these images somewhere. 

    # ignoring checkpointing work - we're training from scratch

    # smart crop - ignore for now, cut scope, etc. 

    # training

def train():




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_data', type=str, default=None)
    parser.add_argument('--instance_data', type=str, default=None)
    parser.add_argument('--captions_data', type=str, default=None)
    args = parser.parse_args()

    cog_instance_data = Path("instance_data")
    cog_class_data = Path("class_data")

    # extract zip contents, flattening any paths present within it
    with ZipFile(args.instance_data, "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                "__MACOSX"
            ):
                continue
            mt = mimetypes.guess_type(zip_info.filename)
            if mt and mt[0] and mt[0].startswith("image/"):
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, cog_instance_data)

    if args.class_data is not None:
        with ZipFile(str(args.class_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_class_data)

