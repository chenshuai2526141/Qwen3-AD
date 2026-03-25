"""
不加载模型、不训练：检查 train.jsonl -> train_sft.jsonl 转换与训练拼接文本是否符合预期。
逻辑与 train_lora.py 中 PROMPT、dataset_jsonl_transfer、process_func 拼接方式保持一致。

用法（在项目根目录）：
  python preview_sft_data.py
  python preview_sft_data.py 3
可选参数为打印条数，默认 2。
"""

import json
import os
import sys

# 与 train_lora.py 保持一致；若修改训练脚本中的提示语，请同步此处
PROMPT = "你是一个广告专家，请根据用户的关键词生成广告宣传语。"


def dataset_jsonl_transfer(origin_path, new_path):
    """与 train_lora.dataset_jsonl_transfer 相同。"""
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            keywords = data["content"]
            ad_text = data["summary"]
            messages.append({
                "instruction": PROMPT,
                "input": keywords,
                "output": ad_text,
            })
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def main():
    n = 2
    if len(sys.argv) > 1:
        try:
            n = max(1, int(sys.argv[1]))
        except ValueError:
            print("用法: python preview_sft_data.py [条数，默认2]")
            sys.exit(1)

    train_dataset_path = "train.jsonl"
    train_jsonl_new_path = "train_sft.jsonl"

    if not os.path.isfile(train_dataset_path):
        print(f"未找到 {train_dataset_path}，请先运行 data.py 生成数据。")
        sys.exit(1)
    if not os.path.isfile(train_jsonl_new_path):
        print(f"生成 {train_jsonl_new_path} ...")
        dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)

    print(f"从 {train_jsonl_new_path} 读取前 {n} 条：\n")
    with open(train_jsonl_new_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            print(f"========== 样本 {i + 1} ==========")
            print("instruction / input / output（JSON）：")
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            assembled = (
                f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{rec['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{rec['output']}"
            )
            print("\n与 train_lora.process_func 中拼接一致的训练文本（未分词、未截断）：")
            print(assembled)
            print()


if __name__ == "__main__":
    main()
