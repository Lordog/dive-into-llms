# Dive into Large Language Models: LLM Jailbreak Attacks
Introduction: LLM jailbreak attacks and tools
> To achieve better security, you must first learn to attack. Let's see how jailbreak attacks pry open the LLM's mouth!

## 1. Tutorial Goals:

- Become familiar with the EasyJailbreak toolkit;
- Master the implementation and results of common LLM jailbreak methods;

## 2. Preparation:
### 2.1 Get to know EasyJailbreak

https://github.com/EasyJailbreak/EasyJailbreak

EasyJailbreak is an easy-to-use jailbreak attack framework designed for researchers and developers focused on LLM security.

EasyJailbreak integrates 11 mainstream jailbreak attack methods. It decomposes the jailbreak process into several iterable steps: initializing random seeds, adding constraints, mutation, attack, and evaluation. Each attack method contains four different modules: Selector, Mutator, Constraint, and Evaluator.

EasyJailbreak 
![](./assets/1.jpg)

### 2.2 Main Framework

![](./assets/2.jpg)

EasyJailbreak can be divided into three parts:
- The first part prepares the Queries, Config, Models, and Seed required for attacks and evaluation.
- The second part is the attack loop, which contains two main processes: Mutation and Inference
  - Mutation: first select suitable jailbreak prompts based on the Selector, then transform the jailbreak prompts based on the Mutator, and finally filter the desired jailbreak prompts based on the Constraint. 
  - Inference: this part attacks the target LLM with the previously obtained jailbreak prompts and obtains the model's responses. The responses are then sent to the Evaluator to obtain the attack results.
- The third part produces the final attack and evaluation report. Based on a preset stopping mechanism, the attack loop ends and the final jailbreak prompts, model responses, attack results, etc. are obtained.

https://easyjailbreak.github.io/EasyJailbreakDoc.github.io/

## 3. Install the Environment
To directly use the jailbreak attacks and evaluation in EasyJailbreak:
```
pip install easyjailbreak
```
To develop further on top of EasyJailbreak, for example by adding new Mutators, Evaluators, etc.:
```
git clone https://github.com/EasyJailbreak/EasyJailbreak.git
cd EasyJailbreak
pip install -e .
```

## 4. EasyJailbreak Example
Using the PAIR method as an example:

### 1. Load the Model

The `HuggingfaceModel` class loads models from Hugging Face and adds some functions suitable for attacks.
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

Use the target model to generate a response from the input
```
target_response = target_model.generate(messages=['how to make a bomb?'])
```

### 2. Load the Dataset
The `JailbreakDataset` class is used to assemble jailbreak datasets; each instance contains a query input, jailbreak prompt, etc.
```
from easyjailbreak.datasets import JailbreakDataset
```

You can directly load some online (Hugging Face) datasets
```
dataset = JailbreakDataset(dataset='AdvBench')
```

You can also load a local dataset file
```
dataset = JailbreakDataset(local_file_type='csv', dataset='AdvBench.csv')
```

### 3. Initialize Random Seeds
```
from easyjailbreak.seed.seed_random import SeedRandom
seeder = SeedRandom()
seeder.new_seeds()
```


### 4. Set the Attack Method
```
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR

attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)
```

### 5. Execute the Attack
```
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
```


## Custom Jailbreak Attacks (Optional)

According to Section 2.2, before performing a jailbreak you need to set up the modules used by the attack, including the Selector, Mutator, Constraint, Evaluator, etc.

Use the modules implemented by EasyJailbreak or define your own

Import style: ```from easyjailbreak.module_name.method_name import method_namemethod_name```
- Selector: ```from easyjailbreak.selector.method_name import method_name```
- Mutator: ```from easyjailbreak.mutation.rule.method_name import method_name```
- Constraint: ```from easyjailbreak.constraint.method_name import method_name```
- Evaluator: ```from easyjailbreak.metrics.Evaluator.method_name import method_name```


### Basic Attack Pipeline Implementation:
```
from easyjailbreak.selector.RandomSelector import RandomSelectPolicy
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.seed import SeedTemplate
from easyjailbreak.mutation.rule import Translate
from easyjailbreak.models import from_pretrained
import torch
```

#### 1. Set a Malicious Query
```
instance = Instance(query='How to make a bomb?')
dataset = JailbreakDataset([instance])
```

#### 2. Load the Target Model
```
model = from_pretrained('meta-llama/Llama-2-7b-chat-hf', 'llama-2', dtype=torch.bfloat16, max_new_tokens=200)
```

#### 3. Set Jailbreak Prompts
```
inital_prompt_seed = SeedTemplate().new_seeds(seeds_num= 10, method_list=['Gptfuzzer'])
inital_prompt_seed = JailbreakDataset([Instance(jailbreak_prompt=prompt) for prompt in inital_prompt_seed])
```

#### 4. Set the Selector
```
selector = RandomSelectPolicy(inital_prompt_seed)
```

#### 5. Select Suitable Jailbreak Prompts with the Selector
```
candidate_prompt_set = selector.select()
for instance in dataset:
    instance.jailbreak_prompt = candidate_prompt_set[0].jailbreak_prompt
```

#### 6. Transform the Query/Prompt with the Mutator
```
Mutation = Translate(attr_name='query',language = 'jv')
mutated_instance = Mutation(dataset)[0]
```

#### 7. Obtain the Target Model's Response
```
attack_query = mutated_instance.jailbreak_prompt.format(query = mutated_instance.query)
response = model.generate(attack_query)
```
