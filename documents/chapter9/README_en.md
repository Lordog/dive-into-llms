# Dive into Large Language Models: GUI Agent Construction
## Tutorial Goals
1. Understand the technical roadmap and research status of the GUI agent field

2. Try the method of building an adaptive human-computer interaction GUI agent based on the open-source model Qwen2-VL-7B and the OS-Kairos dataset

3. Try running inference with the constructed GUI agent
## 1. Preparation
### 1.1 Understand the technical roadmap and research status of the GUI agent field
Read the tutorial: [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter8/GUIagent.pdf)]

### 1.2 Understand what an adaptive human-computer interaction GUI agent is
Reference paper: [[Paper](https://arxiv.org/abs/2503.16465)]

### 1.3 Dataset Preparation
This tutorial uses OS-Kairos as an example and builds and runs inference for a simple GUI agent based on the dataset released by this work.

Download the dataset from the download link in the README.md file of [[Data](https://github.com/Wuzheng02/OS-Kairos)] and extract it into your environment.

An example of the data format is as follows:
```json
    {
        "task": "Open NetEase Cloud Music, search for 'Shape of You', and play this song.",
        "image_path": "/data1/wuzh/cloud_music/images/1736614680.6518524_1.png",
        "list": [
            " Open NetEase Cloud Music  ",
            " Tap the search box at the top of the home page  ",
            " Enter: Shape of You  ",
            " Select the correct search result  ",
            " Tap the song title to play  "
        ],
        "now_step": 1,
        "previous_actions": [
            "CLICK <point>[[381,367]]</point>"
        ],
        "score": 5,
        "osatlas_action": "CLICK <point>[[454,87]]</point>",
        "teacher_action": "CLICK <point>[[500,100]]</point>",
        "success": false
    },
```
### 1.4 Model Preparation
Download the Qwen2-VL-7B model weights from [[Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)].

## 1.5 Supervised Fine-Tuning Code Preparation
This tutorial uses LLaMA-Factory to perform supervised fine-tuning on Qwen2-VL-7B, so first download the source code from [[Code](https://github.com/hiyouga/LLaMA-Factory/)].

## 2. GUI Agent Construction
### 2.1 Data Preprocessing
Save the following code as get_sharpgpt.py, fill in the paths to Kairos_train.json and the JSON file where you plan to store the processed training set, and then run the file.
```python
import json


with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)


preprocessed_data = []
for item in data:
    task = item.get("task", "")
    previous_actions = item.get("previous_actions", "") 
    image_path = item.get("image_path","")
    prompt_text = f"""
    You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>

    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - Purpose: SCROLL in the specified direction.
        - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - Example Usage: SCROLL [UP]

    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

    Custom Action 1: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 2: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 3: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER

    Custom Action 4: IMPOSSIBLE
        - purpose: Indicate the task is impossible.
        - format: IMPOSSIBLE
        - example usage: IMPOSSIBLE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.

    And I hope you evaluate your action to be scored, giving it a score from 1 to 5. 
    A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.
    1 means you believe this action definitely cannot achieve the goal.
    2 means you believe this action is very unlikely to achieve the goal.
    3 means you believe this action has a certain chance of achieving the goal.
    4 means you believe this action is very likely to achieve the goal.
    5 means you believe this action will definitely achieve the goal.

    And your final goal, previous actions, and associated screenshot are as follows:
    Final goal: {task}
    previous actions: {previous_actions}
    screenshot:<image>
    Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas action must be one of the action formats I provided and score must be 1 to 5):
    action:
    score:
    """

    action = item.get("osatlas_action", "")
    score = item.get("score", "")
    preprocessed_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": f"action: {action}\nscore:{score}",
            }
        ],
        "images":[image_path]
    }
    preprocessed_data.append(preprocessed_item)


with open('', 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
```
After this step, we have successfully converted the OS-Kairos training set into data that conforms to the sharpgpt format, facilitating the next training step. Data preprocessing is complete.
## 2.2 Supervised Fine-Tuning
In the previous step, we obtained the Kairos dataset in a format suitable for LLaMA-Factory training. Next, we need to modify LLaMA-Factory to register the dataset and configure the training information.
First, modify data/dataset_info.json and add the following content to register the dataset:
```json
"Karios" :{
  "file_name": "Karios_qwenscore.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
},   
```
Then modify /examples/train_full/qwen2vl_full_sft.yaml to configure the training information:
```yaml
### model
model_name_or_path: 
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: Karios
template: qwen2_vl
cutoff_len: 4096
max_samples: 9999999
#max_samples: 999
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: 
logging_steps: 10
save_steps: 60000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 20000
```
The above is an example configuration. model_name_or_path is the path to Qwen2-VL-7B, and output_dir is the path for storing checkpoints and logs.
Once configured, you can configure inference. Example command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml
```
At least three 80GB A100 GPUs are required here.

## 3. OS-Kairos Inference Validation
After training is complete, we can run inference. There are many inference methods to choose from: you can build inference code yourself using the transformers and torch libraries, or use LLaMA-Factory for one-click inference. The following demonstrates how to use LLaMA-Factory for one-click inference.
First, modify /examples/inference/qwen2_vl.yaml. An example is as follows
```yaml
model_name_or_path: 
template: qwen2_vl
```
Here, model_name_or_path is the path to the checkpoint after training.
Then you can run it with
```
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli webchat examples/inference/qwen2_vl.yaml
```
to perform one-click inference. You can use an image from the OS-Kairos test set or a screenshot from your own phone, combine it with a text prompt in the format used in get_sharpgpt.py, and feed it to the agent. After inference, OS-Kairos will output the action it believes should be taken at the moment and a confidence score for that action. The lower the confidence score, the more the current instruction exceeds the capability boundary of OS-Kairos, requiring human intervention for human-computer interaction.