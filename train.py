import os
from subprocess import getoutput
import time
import random


Resume_Training = False #@param {type:"boolean"}

# if resume and not Resume_Training:
#   print('[1;31mOverwrite your previously trained model ? answering "yes" will train a new model, answering "no" will resume the training of the previous model?  yes or no ?[0m')
#   while True:
#     ansres=input('')
#     if ansres=='no':
#       Resume_Training = True
#       break
#     elif ansres=='yes':
#       Resume_Training = False
#       resume= False
#       break

# while not Resume_Training and MODEL_NAME=="":
#   print('[1;31mNo model found, use the "Model Download" cell to download a model.')
#   time.sleep(5)

#@markdown  - If you're not satisfied with the result, check this box, run again the cell and it will continue training the current model.

MODELT_NAME="/src/content/stable-diffusion-v1-5"

UNet_Training_Steps=1500 
UNet_Learning_Rate = 2e-6 
#["2e-5","1e-5","9e-6","8e-6","7e-6","6e-6","5e-6", "4e-6", "3e-6", "2e-6"] {type:"raw"}
untlr=UNet_Learning_Rate
# These default settings are for a dataset of 10 pictures which is enough for training a face, start with 500 
# or lower, test the model, if not enough, resume training for 150 steps, keep testing until you get the 
# desired output, `set it to 0 to train only the text_encoder`.

Text_Encoder_Training_Steps=350 #@param{type: 'number'}
# 200-450 steps is enough for a small dataset, keep this number small to avoid overfitting, set to 0 to disable, 
# `set it to 0 before resuming training if it is already trained`.

Text_Encoder_Concept_Training_Steps=200
# Suitable for training a style/concept as it acts as heavy regularization, set it to 1500 steps 
# for 200 concept images (you can go higher), set to 0 to disable, set both the settings 
# above to 0 to fintune only the text_encoder on the concept, 
# `set it to 0 before resuming training if it is already trained`.

Text_Encoder_Learning_Rate = 1e-6 
# ["2e-6", "1e-6","8e-7","6e-7","5e-7","4e-7"]
txlr=Text_Encoder_Learning_Rate

# Learning rate for both text_encoder and concept_text_encoder, keep it low to avoid overfitting
# (1e-6 is higher than 4e-7) 

# sets if we're only training the text encoder
trnonltxt=""
if UNet_Training_Steps==0:
   trnonltxt="--train_only_text_encoder"

Seed=''

# TODO: what is this param and why do we care? 
ofstnse=""
Offset_Noise = False #@param {type:"boolean"}
#It should help with the overall quality, when used, increase the amount of the total steps by a 10-20%.
if Offset_Noise:
  ofstnse="--offset_noise"

# TODO: basicalluy this justimplies that captions are in separate .txt files instead of in the filenames of the jpgs themselves
External_Captions = True #@param {type:"boolean"}
#@markdown - Get the captions from a text file for each instance image.
extrnlcptn=""
if External_Captions:
  extrnlcptn="--external_captions"

# TODO: what is this param and why do we care?
Style_Training = False 
#Further reduce overfitting, suitable when training a style or a general theme, don't check the box at the beginning, check it after training for at least 1000 steps. (Has no effect when using External Captions)

Style=""
if Style_Training:
  Style="--Style"

# TODO: assume this is square, is that correct, do we need flexibility, etc. 
Resolution = "512" 
# ["512", "576", "640", "704", "768", "832", "896", "960", "1024"]
Res=int(Resolution)
# Higher resolution = Higher quality, make sure the instance images are cropped to this selected size (or larger).

fp16 = True

if fp16:
  prec="fp16"
else:
  prec="no"

if Seed =='' or Seed=='0':
  Seed=random.randint(1, 999999)
else:
  Seed=int(Seed)

# TODO - needed? 
GC="--gradient_checkpointing"


# TODO - unsure what this GCUNET param is doing. 
s = getoutput('nvidia-smi')
if 'A100' in s:
  GC=""
GCUNET=GC
if Res<=640:
  GCUNET=""

precision=prec

# resuming=""
# if Resume_Training and os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
#   MODELT_NAME=OUTPUT_DIR
#   print('[1;32mResuming Training...[0m')
#   resuming="Yes"
# elif Resume_Training and not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
#   print('[1;31mPrevious model not found, training a new model...[0m')
#   MODELT_NAME=MODEL_NAME
#   while MODEL_NAME=="":
#     print('[1;31mNo model found, use the "Model Download" cell to download a model.')
#     time.sleep(5)

Enable_text_encoder_training= True
Enable_Text_Encoder_Concept_Training= True

if Text_Encoder_Training_Steps==0 :
   Enable_text_encoder_training= False
else:
  stptxt=Text_Encoder_Training_Steps

if Text_Encoder_Concept_Training_Steps==0:
   Enable_Text_Encoder_Concept_Training= False
else:
  stptxtc=Text_Encoder_Concept_Training_Steps

#@markdown ---------------------------
Save_Checkpoint_Every_n_Steps = False #@param {type:"boolean"}
Save_Checkpoint_Every=500 #@param{type: 'number'}
if Save_Checkpoint_Every==None:
  Save_Checkpoint_Every=1
#@markdown - Minimum 200 steps between each save.
stp=0
Start_saving_from_the_step=500 #@param{type: 'number'}
if Start_saving_from_the_step==None:
  Start_saving_from_the_step=0
if (Start_saving_from_the_step < 200):
  Start_saving_from_the_step=Save_Checkpoint_Every
stpsv=Start_saving_from_the_step
if Save_Checkpoint_Every_n_Steps:
  stp=Save_Checkpoint_Every
#@markdown - Start saving intermediary checkpoints from this step.

Disconnect_after_training=False #@param {type:"boolean"}

#@markdown - Auto-disconnect from google colab after the training to avoid wasting compute units.

def dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
    launch_command = f"accelerate launch /src/diffusers/examples/dreambooth/train_dreambooth.py \
        {trnonltxt} {extrnlcptn} {ofstnse} --image_captions_filename --train_text_encoder --dump_only_text_encoder  --pretrained_model_name_or_path={MODELT_NAME} \
        --instance_data_dir={INSTANCE_DIR} --output_dir={OUTPUT_DIR} --instance_prompt={PT} --seed={Seed} --resolution=512 \
        --mixed_precision={precision} --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate={txlr} \
        --lr_scheduler=linear --lr_warmup_steps=0 --max_train_steps={Training_Steps}"
    os.system(launch_command)

    # !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    # $trnonltxt \
    # $extrnlcptn \
    # $ofstnse \
    # --image_captions_filename \ # TODO: this
    # --train_text_encoder \
    # --dump_only_text_encoder \
    # --pretrained_model_name_or_path="$MODELT_NAME" \
    # --instance_data_dir="$INSTANCE_DIR" \
    # --output_dir="$OUTPUT_DIR" \
    # --instance_prompt="$PT" \
    # --seed=$Seed \
    # --resolution=512 \
    # --mixed_precision=$precision \
    # --train_batch_size=1 \
    # --gradient_accumulation_steps=1 $GC \
    # --use_8bit_adam \
    # --learning_rate=$txlr \
    # --lr_scheduler="linear" \
    # --lr_warmup_steps=0 \
    # --max_train_steps=$Training_Steps

def train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps):
    print('Training Unet only...')
    launch_command = f"accelerate launch /src/diffusers/examples/dreambooth/train_dreambooth.py \
        {Style} {extrnlcptn} {ofstnse} --image_captions_filename --train_only_unet --save_starting_step={stpsv} \
        --save_n_steps={stp} --Session_dir={SESSION_DIR} --pretrained_model_name_or_path={MODELT_NAME} \
        --instance_data_dir={INSTANCE_DIR} --output_dir={OUTPUT_DIR} --captions_dir={CAPTIONS_DIR} --instance_prompt={PT} \
        --seed={Seed} --resolution={Res} --mixed_precision={precision} --train_batch_size=1 \
        --gradient_accumulation_steps=1 --learning_rate={untlr} --lr_scheduler=linear --lr_warmup_steps=0 \
        --max_train_steps={Training_Steps}"
    os.system(launch_command)
    # !accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
    # $Style \
    # $extrnlcptn \
    # $ofstnse \
    # --image_captions_filename \
    # --train_only_unet \
    # --save_starting_step=$stpsv \
    # --save_n_steps=$stp \
    # --Session_dir=$SESSION_DIR \
    # --pretrained_model_name_or_path="$MODELT_NAME" \
    # --instance_data_dir="$INSTANCE_DIR" \
    # --output_dir="$OUTPUT_DIR" \
    # --captions_dir="$CAPTIONS_DIR" \
    # --instance_prompt="$PT" \
    # --seed=$Seed \
    # --resolution=$Res \
    # --mixed_precision=$precision \
    # --train_batch_size=1 \
    # --gradient_accumulation_steps=1 $GCUNET \
    # --use_8bit_adam \
    # --learning_rate=$untlr \
    # --lr_scheduler="linear" \
    # --lr_warmup_steps=0 \
    # --max_train_steps=$Training_Steps

Session_Name="test_db/test_dan"

INSTANCE_NAME=Session_Name
OUTPUT_DIR=os.path.join("/src/outputs/",Session_Name,str(time.time()))
os.makedirs(OUTPUT_DIR, exist_ok=True)
SESSION_DIR=os.path.join("/src",Session_Name)
INSTANCE_DIR=os.path.join(SESSION_DIR,'instance_images')
CONCEPT_DIR=os.path.join(SESSION_DIR, 'concept_images')
CAPTIONS_DIR=os.path.join(SESSION_DIR, 'captions')
MDLPTH=str(SESSION_DIR+"/"+Session_Name+'.ckpt')

INSTANCE_DIR # where are the images of the target
INSTANCE_NAME # output dir
PT="" # I think this is empty
SESSION_DIR # 
CONCEPT_DIR

if Enable_text_encoder_training :
  print("training text encoder")
#   if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
#     %rm -r $OUTPUT_DIR"/text_encoder_trained"
  dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxt)

if Enable_Text_Encoder_Concept_Training:
  if os.path.exists(CONCEPT_DIR):
    if os.listdir(CONCEPT_DIR)!=[]:
    #   if resuming=="Yes":
    #     print('[1;32mResuming Training...[0m')
      print("training text encoder on concept images")
      dump_only_textenc(trnonltxt, MODELT_NAME, CONCEPT_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxtc)
    else:
    #   if resuming=="Yes":
    #     print('[1;32mResuming Training...[0m')
      print('config wanted to train text enc but no concept images found, skipping concept training...')
      Text_Encoder_Concept_Training_Steps=0
  else:
    #   clear_output()
    #   if resuming=="Yes":
    #     print('[1;32mResuming Training...[0m')
      print('config wanted to train text enc but no concept images found, skipping concept training...')
      Text_Encoder_Concept_Training_Steps=0



if UNet_Training_Steps!=0:
  train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps=UNet_Training_Steps)

if UNet_Training_Steps==0 and Text_Encoder_Concept_Training_Steps==0 and Text_Encoder_Training_Steps==0 :
  print('[1;32mNothing to do')
else:
  # and here's where we convert the model to a ckpt. 
  path_to_test = os.path.join(OUTPUT_DIR,'unet/diffusion_pytorch_model.bin')
  print(f"checking {path_to_test}")
  if os.path.exists(path_to_test):
    print('found it')
    prc="--fp16" if precision=="fp16" else ""
    save_command = f"python /src/diffusers/scripts/convertosdv2.py {prc} {OUTPUT_DIR} {SESSION_DIR}/{Session_Name}.ckpt"
    os.system(save_command)
    # !python /content/diffusers/scripts/convertosdv2.py $prc $OUTPUT_DIR $SESSION_DIR/$Session_Name".ckpt"
    # clear_output()
    if os.path.exists(os.path.join(SESSION_DIR, (INSTANCE_NAME+'.ckpt'))):
      #clear_output()
      print("Done and saved the CKPT model is in your Gdrive in the sessions folder")
    #   if Disconnect_after_training :
    #     time.sleep(20)
    #     runtime.unassign()
    else:
      print("failed to save the ckpt")
  else:
    print(f"couldn't find anyting at {path_to_test}")
    print("oh nooo something went wrong")