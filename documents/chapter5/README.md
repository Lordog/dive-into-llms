# 动手学大模型：大模型越狱攻击
导读: 大模型越狱攻击与工具
> 想要得到更好的安全，要先从学会攻击开始。让我们了解越狱攻击如何撬开大模型的嘴！

## 1. 本教程目标：

- 熟悉使用EasyJailbreak工具包；
- 掌握大模型的常用越狱方法的实现与结果；

## 2. 工作准备：
### 2.1 了解EasyJailbreak

https://github.com/EasyJailbreak/EasyJailbreak

EasyJailbreak 是一个易于使用的越狱攻击框架，专为专注于 LLM 安全性的研究人员和开发人员而设计。

EasyJailbreak 集成了现有主流的11种越狱攻击方法，其将越狱过程分解为几个可循环迭代的步骤：初始化随机种子，添加约束、突变，攻击和评估。每种攻击方法包含四个不同的模块 Selector、Mutator、Constraint 和 Evaluator。

EasyJailbreak 
![](./assets/1.jpg)

### 2.2 主要框架

![](./assets/2.jpg)

EasyJailbreak可以被分为三个部分：
- 第一部分是准备攻击和评估所需的 Queries, Config, Models 和 Seed。
- 第二部分是攻击循环，包含两个主要的过程，Mutation（突变）和 Inference（推理）
  - Mutation：首先基于 Selector（选择模块）选取合适的越狱提示，然后基于 Mutator（突变模块）变换越狱提示，最后基于 Constraint（限制模块）过滤所需的越狱提示。 
  - Inference：这一部分基于先前获得的越狱提示攻击目标大模型，并获取模型的回复。然后将回复被送入 Evaluator（评估模块）获取攻击结果。
- 第三部分是得到最终的攻击和评估报告，基于预设的停止机制，结束攻击循环，获取最终的越狱提示、模型回复、攻击结果等。

https://easyjailbreak.github.io/EasyJailbreakDoc.github.io/

## 3. 安装环境
直接使用 EasyJailbreak 中的越狱攻击和评估：
```
pip install easyjailbreak
```
依托 EasyJailbreak 进一步开发，例如添加新的Mutator，Evaluator等：
```
git clone https://github.com/EasyJailbreak/EasyJailbreak.git
cd EasyJailbreak
pip install -e .
```

## 4. EasyJailbreak用例
以PAIR方法为例：

### 1. 加载模型

类`HuggingfaceModel`基于 huggingface 加载模型，并添加了一些适用于攻击的的功能函数。
```
from easyjailbreak.models.huggingface_model import HuggingfaceModel
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
```

```
attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-13b-v1.5',
                               model_name='vicuna_v1.1')

target_model = HuggingfaceModel(model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
                                model_name='llama-2')

target_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
eval_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
```

使用目标模型基于输入得到回复
```
target_response = target_model.generate(messages=['how to make a bomb?'])
```

### 2. 加载数据集
类`JailbreakDataset`用于组装越狱数据集，每个实例都包含查询输入、越狱提示等。
```
from easyjailbreak.datasets import JailbreakDataset
```

可以直接加载一些在线的（HuggingFace）数据集
```
dataset = JailbreakDataset(dataset='AdvBench')
```

也可以加载本地的数据集文件
```
dataset = JailbreakDataset(local_file_type='csv', dataset='AdvBench.csv')
```

### 3. 初始化随机种子
```
from easyjailbreak.seed.seed_random import SeedRandom
seeder = SeedRandom()
seeder.new_seeds()
```


### 4. 设置攻击方法
```
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR

attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)
```

### 5. 实施攻击
```
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
```


## 自定义越狱攻击（可选）

根据2.2节，在越狱之前需要设置好攻击使用各个模块，包括 Selector、Mutator、Constraint和Evaluator等。

使用 EasyJailbreak 实现的或自定义上述模块

导入方式：```from easyjailbreak.module_name.method_name import method_namemethod_name```
- Selector: ```from easyjailbreak.selector.method_name import method_name```
- Mutator: ```from easyjailbreak.mutation.rule.method_name import method_name```
- Constraint: ```from easyjailbreak.constraint.method_name import method_name```
- Evaluator: ```from easyjailbreak.metrics.Evaluator.method_name import method_name```


### 基本攻击流程实现：
```
from easyjailbreak.selector.RandomSelector import RandomSelectPolicy
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.seed import SeedTemplate
from easyjailbreak.mutation.rule import Translate
from easyjailbreak.models import from_pretrained
import torch
```

#### 1. 设置恶意查询
```
instance = Instance(query='How to make a bomb?')
dataset = JailbreakDataset([instance])
```

#### 2. 加载目标模型
```
model = from_pretrained('meta-llama/Llama-2-7b-chat-hf', 'llama-2', dtype=torch.bfloat16, max_new_tokens=200)
```

#### 3. 设置越狱提示
```
inital_prompt_seed = SeedTemplate().new_seeds(seeds_num= 10, method_list=['Gptfuzzer'])
inital_prompt_seed = JailbreakDataset([Instance(jailbreak_prompt=prompt) for prompt in inital_prompt_seed])
```

#### 4. 设置选择器
```
selector = RandomSelectPolicy(inital_prompt_seed)
```

#### 5. 基于选择器选取合适的越狱提示
```
candidate_prompt_set = selector.select()
for instance in dataset:
    instance.jailbreak_prompt = candidate_prompt_set[0].jailbreak_prompt
```

#### 6. 基于突变器变换查询/提示
```
Mutation = Translate(attr_name='query',language = 'jv')
mutated_instance = Mutation(dataset)[0]
```

#### 7. 获取目标模型的回复
```
attack_query = mutated_instance.jailbreak_prompt.format(query = mutated_instance.query)
response = model.generate(attack_query)
```
