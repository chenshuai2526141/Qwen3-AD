import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download
import os

def predict(messages, model, tokenizer,device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 定义模型名称
model_name = "Qwen/Qwen3-0.6B"

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "models")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_name, cache_dir=cache_path, revision="master")

# 加载原下载路径的tokenizer和model（CPU 推理）
load_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=load_dtype)
# 优先使用GPU，否则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印一下，方便你看当前用的是什么设备
print(f"使用设备: {device}")
model.to(device)

# 加载lora模型（请将路径改为你实际的 LoRA 输出目录，需包含 adapter_config.json）
# 示例：model = PeftModel.from_pretrained(model, model_id="./output/your_lora_dir")
# 当前占位路径会报错，如未训练 LoRA 请注释下一行
# model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-0.6B/checkpoint-1084")

test_texts = {
    'instruction': "你是一个广告专家，请根据用户的关键词生成广告宣传语。",
    'input': "上衣,显瘦,蓝色,运动,开衫,拉链,拉链"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer,device)
print(response)