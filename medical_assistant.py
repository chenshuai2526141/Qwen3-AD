#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品文案助手（基于 train.jsonl 数据形态）
根据关键词生成电商商品描述，交互逻辑对齐医疗助手脚本；示例与分类来自 train.jsonl。
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 与 inference_lora.py 一致的任务设定
BASE_INSTRUCTION = "你是一个广告专家，请根据用户的关键词生成广告宣传语。"

# 不同写作侧重（对应医疗脚本里的多场景 system 变体）
STYLE_PROMPTS = {
    "tops": f"{BASE_INSTRUCTION} 当前品类：上衣。请突出版型、面料与穿搭场景。",
    "dress": f"{BASE_INSTRUCTION} 当前品类：裙装。请突出廓形、细节与气质。",
    "pants": f"{BASE_INSTRUCTION} 当前品类：裤装。请突出版型、修饰效果与舒适度。",
    "standard": BASE_INSTRUCTION,
    "concise": f"{BASE_INSTRUCTION} 请用两到三句简短有力的文案，适合主图文案。",
    "promo": f"{BASE_INSTRUCTION} 请突出卖点与促销感，语气积极，适合活动页。",
    "detail": f"{BASE_INSTRUCTION} 请写一段细节丰富、适合详情页的长文案。",
    "seo": f"{BASE_INSTRUCTION} 请在自然语句中融入关键词，兼顾可读与搜索相关表达。",
    "story": f"{BASE_INSTRUCTION} 请带一点生活场景与情绪，让文案更有画面感。",
    "compare": f"{BASE_INSTRUCTION} 请用对比或递进方式强调优势（如显瘦、舒适、百搭等）。",
}

PRODUCT_SCENARIOS = {
    "1": "上衣文案（数据：上衣）",
    "2": "裙装文案（数据：裙）",
    "3": "裤装文案（数据：裤）",
    "4": "标准广告文案",
    "5": "简短主图文案",
    "6": "促销活动风",
    "7": "详情页长文案",
    "8": "SEO 友好",
    "9": "场景故事感",
    "10": "对比递进卖点",
}

SCENARIO_KEYS = list(STYLE_PROMPTS.keys())  # 顺序与 PRODUCT_SCENARIOS 1-10 对齐


def load_examples_from_jsonl(jsonl_path, max_per_category=5):
    """从 train.jsonl 按品类收集示例（content / summary）。"""
    buckets = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (obj.get("content") or "").strip()
            if not content:
                continue
            category = content.split(",")[0].strip() or "其他"
            if len(buckets[category]) >= max_per_category:
                continue
            buckets[category].append(
                {"content": content, "summary": (obj.get("summary") or "").strip()}
            )
    return dict(buckets)


class CopywritingAssistant:
    def __init__(self, checkpoint_path, train_jsonl_path):
        self.checkpoint_path = checkpoint_path
        self.train_jsonl_path = train_jsonl_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.examples_by_category = {}
        if os.path.isfile(train_jsonl_path):
            self.examples_by_category = load_examples_from_jsonl(train_jsonl_path)
        else:
            print(f"警告: 未找到数据文件 {train_jsonl_path}，将无法展示来自 train.jsonl 的示例。")

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
        print(f"模型加载完成，设备: {self.device}")

    def predict(self, messages, max_new_tokens=512):
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = (
            inputs.attention_mask.to(model_device)
            if hasattr(inputs, "attention_mask")
            else None
        )
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        new_tokens = generated[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def ask_keywords(self, keywords, style_key="standard", max_tokens=512):
        if style_key not in STYLE_PROMPTS:
            style_key = "standard"
        messages = [
            {"role": "system", "content": STYLE_PROMPTS[style_key]},
            {"role": "user", "content": keywords},
        ]
        self.conversation_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "style": style_key,
                "keywords": keywords,
                "response": None,
            }
        )
        response = self.predict(messages, max_new_tokens=max_tokens)
        self.conversation_history[-1]["response"] = response
        return response

    def show_scenarios(self):
        print("\n🛍️ 商品文案助手 - 可选场景")
        print("=" * 50)
        for key, value in PRODUCT_SCENARIOS.items():
            print(f"{key:>2}. {value}")
        print("=" * 50)

    def show_samples_for_choice(self, scenario_choice):
        """场景 1–3 时展示 train.jsonl 中对应品类的关键词示例。"""
        category_map = {"1": "上衣", "2": "裙", "3": "裤"}
        cat = category_map.get(scenario_choice)
        if not cat:
            return
        samples = self.examples_by_category.get(cat) or []
        print(f"\n📋 {PRODUCT_SCENARIOS[scenario_choice]} — train.jsonl 中的关键词示例（可作输入）:")
        print("-" * 40)
        if not samples:
            print("（暂无该类示例，请直接输入逗号分隔的关键词）")
        else:
            for i, row in enumerate(samples, 1):
                print(f"{i}. {row['content']}")
        print("-" * 40)
        print("输入示例编号可直接采用该组关键词；或直接粘贴自定义关键词。")

    def interactive_mode(self):
        print("\n🤖 商品文案助手已启动（数据形态与 train.jsonl 一致：关键词 → 文案）")
        print("输入 help 查看帮助，quit 退出")

        while True:
            try:
                self.show_scenarios()
                scenario_choice = input("\n请选择场景 (1-10): ").strip()
                if scenario_choice == "quit":
                    break
                if scenario_choice == "help":
                    self.show_help()
                    continue
                if scenario_choice not in PRODUCT_SCENARIOS:
                    print("无效选择，请重新输入")
                    continue

                idx = int(scenario_choice) - 1
                style_key = SCENARIO_KEYS[idx]

                self.show_samples_for_choice(scenario_choice)

                raw = input(
                    f"\n请输入关键词（与 train.jsonl 中 content 格式一致，逗号分隔）: "
                ).strip()
                if not raw:
                    print("关键词不能为空")
                    continue

                keywords = raw
                if raw.isdigit():
                    category_map = {"1": "上衣", "2": "裙", "3": "裤"}
                    cat = category_map.get(scenario_choice)
                    samples = self.examples_by_category.get(cat) or []
                    n = int(raw)
                    if 1 <= n <= len(samples):
                        keywords = samples[n - 1]["content"]
                        ref = samples[n - 1].get("summary", "")
                        if ref:
                            print(f"\n（数据中的参考 summary，仅供对照，不送入模型）\n{ref}\n")
                    else:
                        print("示例编号无效，将把你输入的数字当作关键词文本。")

                print("\n正在生成文案...")
                t0 = time.time()
                out = self.ask_keywords(keywords, style_key)
                elapsed = time.time() - t0
                print(f"\n💡 生成结果（耗时 {elapsed:.2f}s）")
                print("=" * 60)
                print(out)
                print("=" * 60)

                cont = input("\n是否继续？(y/n): ").strip().lower()
                if cont in ("n", "no", "否"):
                    break
            except KeyboardInterrupt:
                print("\n\n已退出。")
                break
            except Exception as e:
                print(f"错误: {e}")
                continue

    def show_help(self):
        print("\n帮助")
        print("=" * 50)
        print("1–3：品类与 train.jsonl 中「上衣/裙/裤」样本一致，可看示例编号快速填入关键词。")
        print("4–10：同一关键词，不同 system 风格侧重。")
        print("输入格式与训练数据 content 相同：如「上衣,显瘦,蓝色,运动,开衫,拉链」")
        print("quit 退出")
        print("=" * 50)

    def save_conversation(self, filename=None):
        if not filename:
            filename = f"copywriting_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"对话已保存: {filename}")

    def batch_from_file(self, path):
        """JSON 数组，每项: keywords, scenario 或 style（style 为 STYLE_PROMPTS 的 key）, max_tokens 可选。"""
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        results = []
        for i, item in enumerate(items, 1):
            print(f"处理 {i}/{len(items)}...")
            style = item.get("style") or item.get("scenario") or "standard"
            if style in PRODUCT_SCENARIOS:
                sk = SCENARIO_KEYS[int(style) - 1] if style.isdigit() else style
            else:
                sk = style if style in STYLE_PROMPTS else "standard"
            kw = item.get("keywords") or item.get("question") or item.get("content") or ""
            resp = self.ask_keywords(
                kw, sk, item.get("max_tokens", 512)
            )
            results.append({"keywords": kw, "style": sk, "response": resp})
        out = f"batch_copywriting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"批量完成: {out}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_jsonl = os.path.join(script_dir, "train.jsonl")
    default_ckpt = os.path.join(script_dir, "output", "Qwen3-0.6B", "checkpoint-1084")

    parser = argparse.ArgumentParser(
        description="商品文案助手：train.jsonl 风格关键词 → 模型生成文案"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=default_ckpt,
        help="模型检查点目录（含 tokenizer 与权重）",
    )
    parser.add_argument(
        "--train-jsonl",
        "-t",
        type=str,
        default=default_jsonl,
        help="train.jsonl 路径，用于展示示例",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        type=str,
        help="非交互：直接输入关键词（配合 --style）",
    )
    parser.add_argument(
        "--style",
        "-s",
        type=str,
        default="standard",
        choices=SCENARIO_KEYS,
        help="风格键，与场景 1–10 的 STYLE_PROMPTS 键一致",
    )
    parser.add_argument("--max-tokens", "-m", type=int, default=512)
    parser.add_argument("--batch", "-b", type=str, help="批量 JSON 文件路径")
    parser.add_argument("--save-history", action="store_true")

    args = parser.parse_args()
    assistant = CopywritingAssistant(args.checkpoint, args.train_jsonl)
    assistant.load_model()

    if args.batch:
        assistant.batch_from_file(args.batch)
    elif args.keywords:
        print("=" * 50)
        print(
            assistant.ask_keywords(args.keywords, args.style, args.max_tokens)
        )
        print("=" * 50)
    else:
        assistant.interactive_mode()

    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
