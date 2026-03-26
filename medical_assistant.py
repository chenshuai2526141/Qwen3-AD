#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广告文案生成推理脚本（与项目训练数据 AdvertiseGen 一致）

- 数据格式：train.jsonl / val.jsonl 每行为 {"content": "关键词,逗号分隔", "summary": "参考宣传语"}
- 对话格式：与 train_lora.py、inference_lora.py 相同
  system = 固定 instruction，user = content（关键词串）
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# 与 train_lora.py、preview_sft_data.py、inference_lora.py 保持一致
PROMPT = "你是一个广告专家，请根据用户的关键词生成广告宣传语。"

# 与 train.jsonl 中样本一致，便于对齐测试
SAMPLE_KEYWORDS = [
    "上衣,显瘦,蓝色,运动,开衫,拉链,拉链",
    "上衣,显瘦,针织衫,v领,罗纹袖口,收口,腰带,罗纹",
    "裤,宽松,线条,阔腿裤",
    "裙,背带裙,弧形,纽扣",
]


def build_messages(keywords: str):
    """与训练时 apply_chat_template 输入一致。"""
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": keywords},
    ]


class AdvertiseGenAssistant:
    def __init__(self, checkpoint_path="./output/Qwen3-0.6B/checkpoint-1084"):
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []

    def _select_device_and_dtype(self):
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 12:
                    raise RuntimeError("Unsupported CUDA capability for current PyTorch")
                _ = torch.zeros(1, device="cuda")
                return "cuda", torch.float16
            except Exception:
                pass
        return "cpu", torch.float32

    def load_model(self):
        print("正在加载模型...")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成！使用设备: {self.device}")

    def predict(self, messages, max_new_tokens=512):
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response

    def generate_from_content(self, content: str, max_tokens=512, reference_summary=None):
        """
        根据关键词串（train 中的 content 字段）生成宣传语。
        reference_summary 仅用于记录对照，不参与推理。
        """
        content = (content or "").strip()
        messages = build_messages(content)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "reference_summary": reference_summary,
            "response": None,
        }
        self.conversation_history.append(record)

        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history[-1]["response"] = response
        return response

    def show_sample_keywords(self):
        print("\n示例关键词（与 train.jsonl 风格一致，逗号分隔）:")
        print("-" * 50)
        for i, kw in enumerate(SAMPLE_KEYWORDS, 1):
            print(f"{i}. {kw}")
        print("-" * 50)

    def interactive_mode(self):
        print("\n广告文案生成（AdvertiseGen 对齐）已启动")
        print("输入 'help' 查看帮助，输入 'quit' 退出")

        while True:
            try:
                self.show_sample_keywords()
                choice = input("\n输入序号选用示例，或直接粘贴关键词串: ").strip()
                if choice == "quit":
                    break
                if choice == "help":
                    self.show_help()
                    continue

                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(SAMPLE_KEYWORDS):
                        content = SAMPLE_KEYWORDS[idx - 1]
                    else:
                        print("序号无效，请重试")
                        continue
                else:
                    content = choice

                if not content:
                    print("关键词不能为空")
                    continue

                print("\n生成中...")
                start = time.time()
                response = self.generate_from_content(content)
                elapsed = time.time() - start

                print(f"\n生成结果 (耗时 {elapsed:.2f}s):")
                print("=" * 60)
                print(response)
                print("=" * 60)

                cont = input("\n继续？(y/n): ").strip().lower()
                if cont in ("n", "no", "否"):
                    break

            except KeyboardInterrupt:
                print("\n\n已退出。")
                break
            except Exception as e:
                print(f"错误: {e}")
                continue

    def show_help(self):
        print("\n帮助:")
        print("- 关键词格式与 train.jsonl 的 content 相同，多为英文逗号分隔的标签")
        print("- 单次: python medical_assistant.py -k \"上衣,显瘦,...\"")
        print("- 批量: -b val.jsonl 或 JSON 数组，每条含 content；可有 summary 作参考写入结果")
        print("- 输入 quit 退出")
        print("=" * 50)

    def save_conversation(self, filename=None):
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ad_generation_{ts}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

        print(f"记录已保存: {filename}")

    def _load_batch_items(self, path: str):
        """支持 .jsonl（AdvertiseGen 行）或 .json 数组。"""
        items = []
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    content = row.get("content") or row.get("input", "")
                    items.append(
                        {
                            "content": content,
                            "summary": row.get("summary"),
                            "max_tokens": row.get("max_tokens", 512),
                        }
                    )
            else:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON 批量文件应为数组")
                for row in data:
                    content = row.get("content") or row.get("input", "")
                    items.append(
                        {
                            "content": content,
                            "summary": row.get("summary"),
                            "max_tokens": row.get("max_tokens", 512),
                        }
                    )
        return items

    def batch_from_file(self, batch_path: str):
        try:
            items = self._load_batch_items(batch_path)
        except Exception as e:
            print(f"读取批量文件失败: {e}")
            return

        print(f"共 {len(items)} 条，开始生成...")

        results = []
        for i, it in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] content: {it['content'][:80]}...")
            response = self.generate_from_content(
                it["content"],
                max_tokens=it["max_tokens"],
                reference_summary=it.get("summary"),
            )
            out = {
                "content": it["content"],
                "generated": response,
            }
            if it.get("summary") is not None:
                out["reference_summary"] = it["summary"]
            results.append(out)

        out_name = f"batch_ad_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n完成，结果已写入: {out_name}")


def main():
    parser = argparse.ArgumentParser(
        description="AdvertiseGen 风格推理（与 train.jsonl / train_lora.py 对齐）"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="./output/Qwen3-0.6B/checkpoint-1084",
        help="模型检查点目录",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        type=str,
        dest="content",
        help="关键词串（对应数据中的 content 字段）",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=512,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        help="批量：AdvertiseGen 的 .jsonl 或 JSON 数组文件",
    )
    parser.add_argument("--save-history", action="store_true", help="保存本次运行记录为 JSON")

    args = parser.parse_args()

    assistant = AdvertiseGenAssistant(args.checkpoint)
    assistant.load_model()

    if args.batch:
        assistant.batch_from_file(args.batch)
    elif args.content:
        print("生成结果:")
        print("=" * 50)
        print(assistant.generate_from_content(args.content, max_tokens=args.max_tokens))
        print("=" * 50)
    else:
        assistant.interactive_mode()

    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
