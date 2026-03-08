"""
IGP 数据处理模块

提供数据加载、清洗、转换、验证、格式化等功能的独立模块。
与训练逻辑解耦，专注于数据处理。
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
from datasets import Dataset
from torch.utils.data import DataLoader


class IGPColBERTCollator:
    """
    IGP 专用的 ColBERT Collator
    
    支持 instruction_mask 的处理，用于 IGP 架构中的指令识别训练。
    手动构建 query + instruction 的拼接，并生成对应的 instruction mask。
    
    数据格式要求:
        - query: 查询文本
        - instruction: 指令文本 (可选)
        - positive: 正例文档
        - negative: 负例文档
    """
    
    def __init__(
        self,
        tokenizer,
        max_query_length: int = 32,
    ):
        """
        Args:
            tokenizer: ColBERT 模型的 tokenizer
            max_query_length: query 的最大长度
        """
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.valid_label_columns = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将数据批次转换为模型输入
        
        格式: [CLS] Query [SEP] Instruction [SEP]
        token_labels: 0=Query部分, 1=Instruction部分
        """
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        
        q_ids_list = []
        q_attn_list = []
        q_labels_list = []
        
        for item in features:
            query_text = item.get('query', '')
            instruction_text = item.get('instruction', '')
            
            # 端到端训练：不分割 query，保持完整查询
            # Probe 模块需要自己学习识别指令部分
            # 如果没有单独的 instruction 字段，保持 token_labels 全为 0
            # 模型需要自己学习哪些 token 是指令
            
            q_tokens = self.tokenizer.encode(query_text, add_special_tokens=False)
            
            i_tokens = []
            if instruction_text and instruction_text.strip():
                i_tokens = self.tokenizer.encode(" " + instruction_text.strip(), add_special_tokens=False)
            
            # 构建: [CLS] Query [SEP] Instruction [SEP]
            # Query 部分是 token_labels=0, Instruction 部分是 token_labels=1
            curr_ids = [cls_id]
            curr_ids.extend(q_tokens)
            curr_ids.append(sep_id)  # Query 结束标记
            
            query_end = len(curr_ids)
            curr_ids.extend(i_tokens)
            curr_ids.append(sep_id)  # Instruction 结束标记
            
            # token_labels: query部分=0, instruction部分=1
            curr_labels = [0] * query_end + [1] * (len(i_tokens) + 1)
            
            if len(curr_ids) > self.max_query_length:
                curr_ids = curr_ids[:self.max_query_length]
                curr_labels = curr_labels[:self.max_query_length]
            
            q_ids_list.append(torch.tensor(curr_ids, dtype=torch.long))
            q_labels_list.append(torch.tensor(curr_labels, dtype=torch.float))
            q_attn_list.append(torch.ones(len(curr_ids), dtype=torch.long))
        
        padded_q_ids = torch.nn.utils.rnn.pad_sequence(q_ids_list, batch_first=True, padding_value=pad_id)
        padded_q_attn = torch.nn.utils.rnn.pad_sequence(q_attn_list, batch_first=True, padding_value=0)
        padded_q_labels = torch.nn.utils.rnn.pad_sequence(q_labels_list, batch_first=True, padding_value=0.0)
        
        max_len = padded_q_ids.shape[1]
        padded_q_labels_expanded = torch.zeros_like(padded_q_ids, dtype=torch.float)
        padded_q_labels_expanded[:, :padded_q_labels.shape[1]] = padded_q_labels
        
        positives = [f.get('positive', f.get('pos', [''])[0] if isinstance(f.get('pos'), list) else f.get('pos', '')) for f in features]
        negatives = [f.get('negative', f.get('neg', [''])[0] if isinstance(f.get('neg'), list) else f.get('neg', '')) for f in features]
        
        pos_tokens = self.tokenizer(positives, padding=True, truncation=True, max_length=512, return_tensors='pt')
        neg_tokens = self.tokenizer(negatives, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        # 使用 SentenceTransformerTrainer 期望的格式
        # sentence_0 = query (with instruction), sentence_1 = positive, sentence_2 = negative
        batch = {
            'sentence_0_input_ids': padded_q_ids,
            'sentence_0_attention_mask': padded_q_attn,
            'sentence_0_token_labels': padded_q_labels_expanded,
            
            'sentence_1_input_ids': pos_tokens['input_ids'],
            'sentence_1_attention_mask': pos_tokens['attention_mask'],
            
            'sentence_2_input_ids': neg_tokens['input_ids'],
            'sentence_2_attention_mask': neg_tokens['attention_mask'],
        }
        
        return batch


class DataValidator:
    """数据验证器：检查数据格式和必要字段"""
    
    REQUIRED_FIELDS = ['query', 'pos', 'neg']
    OPTIONAL_FIELDS = ['instruction', 'instruction_mask', 'instruction_text']
    
    @staticmethod
    def validate_item(item: Dict[str, Any]) -> Tuple[bool, str]:
        """验证单条数据"""
        for field in DataValidator.REQUIRED_FIELDS:
            if field not in item:
                return False, f"Missing required field: {field}"
        
        if not item.get('query'):
            return False, "Empty query"
        
        if not item.get('pos'):
            return False, "Empty positive document"
        
        if not item.get('neg'):
            return False, "Empty negative document"
        
        return True, "Valid"
    
    @staticmethod
    def validate_dataset(dataset: Dataset) -> Tuple[List[int], List[str]]:
        """验证整个数据集"""
        valid_indices = []
        errors = []
        
        for idx, item in enumerate(dataset):
            is_valid, msg = DataValidator.validate_item(item)
            if is_valid:
                valid_indices.append(idx)
            else:
                errors.append(f"Index {idx}: {msg}")
        
        return valid_indices, errors
    
    @staticmethod
    def print_dataset_info(dataset: Dataset):
        """打印数据集信息"""
        print(f"\n{'='*60}")
        print(f"📊 Dataset Information")
        print(f"{'='*60}")
        print(f"   Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Available fields: {list(sample.keys())}")
            
            print(f"\n   Sample:")
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"      {key}: {value[:100]}...")
                else:
                    print(f"      {key}: {value}")
        print(f"{'='*60}\n")


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_from_file(
        file_path: str,
        expand_pairs: bool = True,
        validate: bool = True,
    ) -> tuple:
        """加载训练数据
        
        Args:
            file_path: 数据文件路径
            expand_pairs: 是否展开 pairs
            validate: 是否验证数据
            
        Returns:
            (dataset, stats)
        """
        import json
        import random
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    
                    if 'neg' in item and isinstance(item['neg'], str):
                        item['neg'] = [item['neg']]
                    
                    if 'pos' in item and isinstance(item['pos'], str):
                        item['pos'] = [item['pos']]
                    
                    data.append(item)
        
        # 收集所有文档用于随机采样负例
        all_documents = []
        for item in data:
            all_documents.extend(item.get('pos', []))
            all_documents.extend(item.get('neg', []))

        if expand_pairs:
            expanded = []
            for item in data:
                query = item.get('query', '')
                instruction = item.get('instruction', '')
                instruction_substring = item.get('instruction_substring', '')

                positives = item.get('pos', [])
                negatives = item.get('neg', [])

                # 1. 添加带指令的样本（changed）
                # 正例：符合指令的文档
                # 负例：违反指令的文档
                for pos in positives:
                    for neg in negatives:
                        expanded.append({
                            'anchor': query,
                            'positive': pos,
                            'negative': neg,
                            'instruction': instruction,
                            'instruction_substring': instruction_substring,
                        })

                # 2. 添加无指令的样本（original）
                # 去掉指令约束后，原先的 pos 和 neg 都变成正例（都是相关文档）
                # 从其他查询中采样真正的负例
                if len(positives) > 0 and len(negatives) > 0:
                    # 合并所有相关文档作为正例
                    all_relevant_docs = positives + negatives

                    # 从全局文档池中采样负例（排除当前查询的相关文档）
                    current_docs = set(all_relevant_docs)
                    global_neg_candidates = [doc for doc in all_documents if doc not in current_docs]

                    if len(global_neg_candidates) > 0:
                        for rel_doc in all_relevant_docs:
                            # 随机选择一个全局负例
                            random_neg = random.choice(global_neg_candidates)
                            expanded.append({
                                'anchor': query,
                                'positive': rel_doc,
                                'negative': random_neg,
                                'instruction': '',  # 空指令
                                'instruction_substring': '',
                            })

            data = expanded
        
        dataset = Dataset.from_list(data)
        # 确保列顺序符合预期: anchor, positive, negative
        dataset = dataset.select_columns(['anchor', 'positive', 'negative', 'instruction', 'instruction_substring'])
        
        stats = {
            'total_raw': len(data),
            'total_expanded': len(data),
        }
        
        return dataset, stats
    
    @staticmethod
    def add_instruction_masks(
        dataset: Dataset,
        tokenizer,
        max_query_length: int = 32,
    ) -> Dataset:
        """添加 instruction masks
        
        由于现在 collator 已经手动构建 token_labels，这里不再需要预处理
        """
        return dataset


class DataConverter:
    
    @staticmethod
    def convert_to_igp_format(
        query: str,
        positive: str,
        negative: str,
        instruction: str = "",
        instruction_substring: str = "",
    ) -> Dict[str, Any]:
        """转换为 IGP 训练所需的格式"""
        return {
            'query': query,
            'positive': positive,
            'negative': negative,
            'instruction': instruction,
            'instruction_substring': instruction_substring,
        }
    
    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        """加载 JSONL 格式数据"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], path: str):
        """保存为 JSONL 格式"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class DataCleaner:
    """数据清洗器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        
        text = text.strip()
        
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def clean_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗数据集"""
        cleaned = []
        
        for item in dataset:
            cleaned_item = {}
            
            for key, value in item.items():
                if isinstance(value, str):
                    cleaned_item[key] = DataCleaner.clean_text(value)
                else:
                    cleaned_item[key] = value
            
            is_valid, _ = DataValidator.validate_item(cleaned_item)
            if is_valid:
                cleaned.append(cleaned_item)
        
        return cleaned


class DataAugmentor:
    """数据增强器"""
    
    @staticmethod
    def add_instruction_prefix(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为数据添加指令前缀"""
        augmented = []
        
        for item in data:
            new_item = item.copy()
            
            if 'instruction' in item and item['instruction']:
                instruction = item['instruction']
                query = item.get('query', '')
                
                new_item['combined_query'] = f"{instruction} {query}"
            else:
                new_item['combined_query'] = item.get('query', '')
            
            augmented.append(new_item)
        
        return augmented
    
    @staticmethod
    def generate_instruction_mask(
        instruction: str,
        query: str,
        tokenizer,
        max_length: int = 32,
    ) -> List[int]:
        """生成 instruction mask
        
        结构: [CLS] [Q] Query... Instruction...
        Mask:  0    0    0...    1...
        
        Args:
            instruction: 指令文本
            query: 查询文本
            tokenizer: 分词器
            max_length: 最大长度
            
        Returns:
            instruction_mask: 0/1 列表
        """
        cls_id = tokenizer.cls_token_id
        pad_id = tokenizer.pad_token_id
        
        try:
            q_marker_id = tokenizer.convert_tokens_to_ids("[Q]")
        except:
            q_marker_id = None
        
        q_tokens = tokenizer.encode(query, add_special_tokens=False)
        
        i_tokens = []
        if instruction and instruction.strip():
            i_tokens = tokenizer.encode(" " + instruction.strip(), add_special_tokens=False)
        
        curr_ids = [cls_id]
        if q_marker_id:
            curr_ids.append(q_marker_id)
        
        boundary = len(curr_ids)
        curr_ids.extend(q_tokens)
        curr_ids.extend(i_tokens)
        
        curr_labels = [0] * boundary + [1] * len(i_tokens)
        
        if len(curr_ids) > max_length:
            curr_ids = curr_ids[:max_length]
            curr_labels = curr_labels[:max_length]
        
        while len(curr_labels) < max_length:
            curr_labels.append(0)
        
        return curr_labels
