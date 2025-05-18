# 动手学大模型：GUI智能体构建
## 本教程目标
1.了解GUI智能体领域的技术路线和研究现状

2.尝试基于开源模型Qwen2-VL-7B依托OS-Kairos数据集构建自适应人机交互GUI智能体的方法

3.尝试基于构建的GUI智能体进行推理
## 1.准备工作
### 1.1 了解GUI智能体领域的技术路线和研究现状
阅读教程：[[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter8/GUIagent.pdf)]

### 1.2 了解什么是自适应人机交互GUI智能体
参考论文：[[Paper](https://arxiv.org/abs/2503.16465)]

### 1.3 数据集准备
本教程以OS-Kairos为例，基于此工作开源的数据集构建并推理简单的GUI智能体。

从[[Data](https://github.com/Wuzheng02/OS-Kairos)]的README.md文件中的下载链接下载数据集，并解压到环境中。

数据格式示例如下：
```json
    {
        "task": "打开网易云音乐，搜索《Shape of You》，并播放这首歌。",
        "image_path": "/data1/wuzh/cloud_music/images/1736614680.6518524_1.png",
        "list": [
            " 打开网易云音乐  ",
            " 点击首页顶部的搜索框  ",
            " 输入：Shape of You  ",
            " 选择正确的搜索结果  ",
            " 点击歌名以播放  "
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
### 1.4 模型准备
从[[Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)]下载Qwen2-VL-7B模型权重。

## 1.5 有监督微调代码准备
本教程采用LLaMa-Factory对Qwen2-VL-7B进行有监督微调，因此先从[[Code](https://github.com/hiyouga/LLaMA-Factory/)]下载源码。

## 2.GUI智能体构建
### 2.1 数据预处理
将以下代码存储为get_sharpgpt.py，并向其填出Kairos_train.json和预计存储处理后的训练集json的路径，然后运行该文件。
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
经过该步骤，我们将OS-Kairos的训练集成功转化为符合sharpgpt格式的数据，便于进行下一步训练，数据预处理完成。
## 2.2 有监督微调
在上一步中，我们已经得到适配LLaMa-Factory训练的格式的Kairos数据集。然后我们要修改LLaMa-Factory以注册数据集和配置训练信息。
首先修改data/dataset_info.json，添加如下内容注册数据集：
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
继续修改/examples/train_full/qwen2vl_full_sft.yaml来进行配置训练信息：
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
以上是一个示例配置，model_name_or_path是Qwen2-VL-7B的路径，output_dir是存放断点和log的路径。
配置好后即可配置推理，示例命令行：
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml
```
此处至少要使用3张80GB的A100计算资源。

## 3.OS-Kairos推理验证
训练完毕后，我们即可进行推理，推理也有很多种方法可以选择，可以自行通过transformers库和torch库构建推理代码，也可以采用LLaMa-Factory进行一键式推理，以下演示如何LLaMa-Factory进行一键式推理。
首先要修改/examples/inference/qwen2_vl.yaml，一个示例如下
```yaml
model_name_or_path: 
template: qwen2_vl
```
其中model_name_or_path是训练后的断点路径。
然后即可通过
```
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli webchat examples/inference/qwen2_vl.yaml
```
进行一键式推理，你可以自己拿出OS-Kairos测试集的图片，或者个人手机的截图，结合get_sharpgpt.py中格式的文本prompt输入给智能体，OS-Karios推理后会给出它认为当前应该进行的action和对于这个action的置信度分数。置信度分数越低，则当前指令越超出OS-Kairos的能力边界，需要人类介入来进行人机交互。