# Dive into Large Language Models: Multimodal Large Language Models

Introduction: This section introduces common architectures and construction methods for multimodal large language models

> The emergence of large language models has shown that advanced intelligence is fully manifested in the language modality. As multimodal large language models capable of more fully simulating the real world, how can they achieve stronger multimodal understanding and generation capabilities? Can multimodal large language models help realize AGI?


## Tutorial Goals

1. Become familiar with the types of multimodal large language models
2. Master the general technical framework of multimodal large language models
3. Master the construction, training, and inference of multimodal large language models



## 1. Theoretical Background



### 1.1 Understand the Types of Multimodal Large Language Models

- Classification of existing multimodal large language models by functionality and modality support


> Before building multimodal large language models (MLLMs), researchers in this field reached a consensus and a key premise: due to scaling laws and emergent phenomena, current language-based LLMs already possess strong semantic understanding capabilities. This means that language has become the key modality carrying intelligence, so language intelligence is regarded as the hub of multimodal intelligence. Therefore, almost all MLLMs are built on language-based LLMs, with the LLM serving as the core decision-making module, similar to a brain or central processing unit. In other words, by adding additional external non-text modality modules or encoders, the LLM is endowed with multimodal perception and operation capabilities.

We classify existing MLLMs into different types based on their modality and functional support.

![mllm](assets/MLLM-summary.png)


- Read the full tutorial: [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter6/mllms.pdf)] 


- More related surveys
    - [A Survey on Multimodal Large Language Models, https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models, 2023](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
    - [MM-LLMs: Recent Advances in MultiModal Large Language Models, 2023](https://arxiv.org/pdf/2401.13601)


### 1.2 Understand the General Technical Framework of Multimodal Large Language Models


- Architecture 1: LLM as Task Scheduler

> There are currently two common MLLM architectures in the community.
The first is the "LLM as discrete scheduler/controller" architecture. As shown in the figure below, the LLM's role is to receive text signals and issue text commands to downstream modules. All message passing inside the system is mediated by plain-text commands output by the LLM. There is no interaction between different functional modules.


![Architecture1](assets/Architecture1.png)

- Architecture 2: LLM as Joint Part of System

> The second architecture is the encoder-LLM-decoder framework.
This is also the most popular architecture at present. Here the LLM's role is to perceive multimodal information and make responses and operations by itself within an encoder-LLM-decoder structure. Therefore, the key difference from the architecture in Chapter 1 is that the LLM is a key joint part of the system, directly receiving multimodal information from outside and delegating instructions to the decoder/generator in a smoother way. In the encoder-LLM-decoder framework, as shown in the figure below, the encoder processes encoded signals from multiple modalities, the LLM acts as the core decision-maker, and the decoder manages multimodal output.

![Architecture2](assets/Architecture2.png)





## 2. Hands-on Practice with General Multimodal Large Language Models

> Practice the construction process of a general multimodal large language model that supports "any modality to any modality"


### 2.1 A General Unified "Any-to-Any Modality" MLLM: NExT-GPT

> The future of MLLM research will definitely move toward increasingly general generalist models, so they will include as many modalities and functions as possible. NExT-GPT is one of the most pioneering works in this field, introducing the concept of "any-to-any modality" MLLMs for the first time. This architecture achieves powerful functionality and lays the foundation for future research directions in multimodal large language models.

![NExT-GPT](assets/NExT-GPT-screen.png)


> In the code practice portion of this course on multimodal large language models, we will use the NExT-GPT code as the target for accessible analysis and practice.

[NExT-GPT Project](https://next-gpt.github.io/)

[NExT-GPT GitHub repository](https://github.com/NExT-GPT/NExT-GPT)




### 2.2 Code Framework Overview


```
├── figures
├── data
│   ├── T-X_pair_data  
│   │   ├── audiocap                      # text-autio pairs data
│   │   │   ├── audios                    # audio files
│   │   │   └── audiocap.json             # the audio captions
│   │   ├── cc3m                          # text-image paris data
│   │   │   ├── images                    # image files
│   │   │   └── cc3m.json                 # the image captions
│   │   └── webvid                        # text-video pairs data
│   │   │   ├── videos                    # video files
│   │   │   └── webvid.json               # the video captions
│   ├── IT_data                           # instruction data
│   │   ├── T+X-T_data                    # text+[image/audio/video] to text instruction data
│   │   │   ├── alpaca                    # textual instruction data
│   │   │   ├── llava                     # visual instruction data
│   │   ├── T-T+X                         # synthesized text to text+[image/audio/video] instruction data
│   │   └── MosIT                         # Modality-switching Instruction Tuning instruction data
├── code
│   ├── config
│   │   ├── base.yaml                     # the model configuration 
│   │   ├── stage_1.yaml                  # enc-side alignment training configuration
│   │   ├── stage_2.yaml                  # dec-side alignment training configuration
│   │   └── stage_3.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   ├── stage_1.json                  # deepspeed configuration for enc-side alignment training
│   │   ├── stage_2.json                  # deepspeed configuration for dec-side alignment training
│   │   └── stage_3.json                  # deepspeed configuration for instruction-tuning training
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion 
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...  
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration 
│   └── ...
├── ckpt                           
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt         
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...       
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```


### 2.3 Install the Environment

First clone the repository and install the required environment. You can complete the environment setup by running the following commands:

```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```




### 2.4 Getting Started with System Inference


#### 2.4.1 Load the Pretrained NExT-GPT Model Checkpoint

- **Step 1**: Load the `frozen parameters`. [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT) is trained based on the following existing models or modules; please prepare the checkpoints according to the instructions below.
    - `ImageBind` is a unified image/video/audio encoder. You can download the pretrained checkpoint from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth), version `huge`. Then place the `imagebind_huge.pth` file in [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/).
    - `Vicuna`: first prepare LLaMA by following the instructions [[here]](ckpt/pretrained_ckpt/prepare_vicuna.md). Then place the pretrained model in [[./ckpt/pretrained_ckpt/vicuna_ckpt/]](ckpt/pretrained_ckpt/vicuna_ckpt/).
    - `Image Diffusion` is used to generate images. NExT-GPT uses version `v1-5` of [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5). (_The code will download it automatically._)
    - `Audio Diffusion` is used to generate audio content. NExT-GPT uses version `l-full` of [AudioLDM](https://github.com/haoheliu/AudioLDM). (_The code will download it automatically._)
    - `Video Diffusion` is used for video generation. We use version `v2_576w` of [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w). (_The code will download it automatically._)

- **Step 2**: Load the `tunable parameters`.

Place the NExT-GPT system in [[./ckpt/delta_ckpt/nextgpt/7b_tiva_v0]](./ckpt/delta_ckpt/nextgpt/7b_tiva_v0). You can choose to 1) use your own trained parameters, or 2) download the pretrained checkpoint from [Huggingface](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0).


#### 2.4.2 Gradio Demo Deployment

After loading the checkpoints, you can run the demo locally as follows:
```angular2html
cd ./code
bash scripts/app.sh
```
The key parameters are specified as follows:
- `--nextgpt_ckpt_path`: path to the pretrained NExT-GPT parameters.



#### 2.4.3 Hands-on Test Examples

The current version supports arbitrary combinations of text, image, video, and audio inputs, as well as task-combined modality outputs.
It also supports multi-turn contextual interaction.

Please run the tests yourself to see the results.

- **Case-1**: input T+I, output T+A

![case1](assets/T+I-T+A.png)


- **Case-2**: input T+V, output T+A

![case2](assets/T+V-T+A.png)


- **Case-3**: input T+I, output T+I+V

![case3](assets/T+I-T+I+V.png)


- **Case-4**: input T, output T+I+V+A

![case4](assets/T-T+I+V+A.png)







### 2.5 System Training Process


#### 2.5.1 Data Preparation

Please download the following datasets for model training:

A) T-X pair data
  - `CC3M` ***text-image*** pair data. Please follow the instructions [[here]](./data/T-X_pair_data/cc3m/prepare.md). Then place the data in [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
  - `WebVid` ***text-video*** pair data. See the [[instruction]](./data/T-X_pair_data/webvid/prepare.md). The files should be saved in [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
  - `AudioCap` ***text-audio*** pair data. See the [[instruction]](./data/T-X_pair_data/audiocap/prepare.md). Save the data in [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Instruction fine-tuning data
  - T+X-T
    - ***Visual instruction data*** (from `LLaVA`); download it from [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) and place it in [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
    - ***Textual instruction data*** (from `Alpaca`); download it from [here](https://github.com/tatsu-lab/stanford_alpaca) and place it in [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
    - ***Video instruction data*** (from `VideoChat`); download it from [here](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) and place it in [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).

    Note: After downloading the dataset, run `preprocess_dataset.py` to preprocess it into a unified format.
  - T-X+T (T2M)
    - The `T-X+T` instruction dataset (T2M) is stored in [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data).
   
  - MosIT
    - See [here](./data/IT_data/MosIT_data/) for instructions on downloading the data. Finally, place it in [[./data/IT_data/MosIT_data/]](./data/IT_data/MosIT_data/).




#### 2.5.2 Embedding Preparation

In the decoder-side alignment training of NExT-GPT, we minimize the distance between the representations of Signal Tokens and captions. To ensure system efficiency and save time and memory costs, we use the text encoders from each diffusion model to precompute the text embeddings of image, audio, and video captions.

Before training NExT-GPT, run the following command. The generated `embedding` files will be saved in [[./data/embed]](./data/embed).
```
cd ./code/
python process_embeddings.py ../data/T-X_pair_data/cc3m/cc3m.json image ../data/embed/ runwayml/stable-diffusion-v1-5
```


Parameter descriptions:
- args[1]: caption file path;
- args[2]: modality, which can be `image`, `video`, or `audio`;
- args[3]: path to save the embedding files;
- args[4]: name of the corresponding pretrained diffusion model.




#### 2.5.3 Three-Stage Training


First refer to the base configuration file [[./code/config/base.yaml]](./code/config/base.yaml) to understand the basic system settings of the entire module.

Then run the following script to start NExT-GPT training:
```
cd ./code
bash scripts/train.sh
```

The specified command is as follows:
```
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model nextgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/\
    --log_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/
```
Key parameters:
- `--include`: `localhost:0` indicates GPU CUDA index `0` in DeepSpeed.
- `--stage`: training stage.
- `--save_path`: directory for storing the trained delta weights. This directory will be created automatically.
- `--log_path`: directory for storing log files.



The entire NExT-GPT training process is divided into 3 steps:

- **Step 1**: Encoder-side LLM-centric multimodal alignment. This stage trains the ***input projection layer*** while freezing ImageBind, the LLM, and the output projection layer.
  
  Simply run the `train.sh` script above and set: `--stage 1`
  
  Also refer to the run configuration file [[./code/config/stage_1.yaml]](./code/config/stage_1.yaml) and the DeepSpeed configuration file [[./code/dsconfig/stage_1.yaml]](./code/dsconfig/stage_1.yaml) for more step-by-step configuration information.

  Note that the datasets used in this step are listed in `dataset_name_list`, and the dataset names must exactly match the definitions in [[./code/dataset/catalog.py]](./code/dataset/catalog.py).



- **Step 2**: Decoder-side instruction-following alignment. This stage trains the ***output projection layer*** while freezing ImageBind, the LLM, and the input projection layer.

  Simply run the `train.sh` script above and set: `--stage 2`

  Also refer to the run configuration file [[./code/config/stage_2.yaml]](./code/config/stage_2.yaml) and the DeepSpeed configuration file [[./code/dsconfig/stage_2.yaml]](./code/dsconfig/stage_2.yaml) for more step-by-step configuration information.



- **Step 3**: Instruction tuning. This stage makes the following adjustments on the instruction dataset: 1) tune the ***LLM*** via LoRA, 2) tune the ***input projection layer***, and 3) tune the ***output projection layer***.

  Simply run the `train.sh` script above and set: `--stage 3`

  Also refer to the run configuration file [[./code/config/stage_3.yaml]](./code/config/stage_3.yaml) and the DeepSpeed configuration file [[./code/dsconfig/stage_3.yaml]](./code/dsconfig/stage_3.yaml) for more step-by-step configuration information.
