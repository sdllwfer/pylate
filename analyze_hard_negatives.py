#!/usr/bin/env python3
"""
分析训练数据集，识别缺乏 hard negatives 的样本
Hard negatives 定义：与 query 语义相关但不符合 instruction 的文档
"""

import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict


def load_dataset(file_path: str) -> List[Dict]:
    """加载数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_keywords(text: str) -> set:
    """提取文本中的关键词（简单实现）"""
    # 转换为小写，提取字母数字单词
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # 过滤常见停用词
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye', 'ago', 'off', 'too', 'any', 'say', 'man', 'try', 'ask', 'end', 'why', 'let', 'put', 'say', 'she', 'try', 'way', 'own', 'say', 'too', 'old', 'tell', 'very', 'when', 'much', 'would', 'there', 'their', 'what', 'said', 'each', 'which', 'will', 'about', 'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'while', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'just', 'like', 'over', 'also', 'back', 'only', 'come', 'make', 'well', 'were', 'time', 'than', 'them', 'into', 'your', 'some', 'more', 'look', 'work', 'life', 'even', 'here', 'take', 'year', 'most', 'long', 'last', 'find', 'give', 'does', 'made', 'part', 'such', 'keep', 'call', 'came', 'need', 'feel', 'seem', 'turn', 'hand', 'high', 'sure', 'upon', 'head', 'help', 'home', 'side', 'both', 'five', 'once', 'same', 'must', 'name', 'left', 'each', 'done', 'open', 'case', 'show', 'live', 'play', 'went', 'told', 'seen', 'hear', 'talk', 'soon', 'read', 'stop', 'face', 'fact', 'land', 'line', 'kind', 'next', 'word', 'came', 'went', 'told', 'seen', 'hear', 'talk', 'soon', 'read', 'stop', 'face', 'fact', 'land', 'line', 'kind', 'next', 'word'}
    return set(w for w in words if w not in stopwords and len(w) > 3)


def calculate_semantic_similarity(query: str, doc: str) -> float:
    """
    计算 query 和 doc 的语义相似度（基于关键词重叠）
    这是一个简化版本，用于识别明显相关或明显无关的文档
    """
    query_keywords = extract_keywords(query)
    doc_keywords = extract_keywords(doc)
    
    if not query_keywords or not doc_keywords:
        return 0.0
    
    # 计算 Jaccard 相似度
    intersection = len(query_keywords & doc_keywords)
    union = len(query_keywords | doc_keywords)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def check_instruction_compliance(doc: str, instruction: str) -> bool:
    """
    检查文档是否符合 instruction 的要求
    这是一个启发式检查，基于关键词匹配和语义分析
    """
    doc_lower = doc.lower()
    instruction_lower = instruction.lower()
    
    # 提取 instruction 中的关键要求
    # 例如："include the name of the drug" -> 检查文档是否包含药物名称
    
    # 检查常见的 instruction 模式
    compliance_indicators = []
    
    # 1. 检查是否要求具体名称/实体
    if 'name of' in instruction_lower or 'identify' in instruction_lower:
        # 检查文档是否包含大写名称（可能是专有名词）
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc))
        compliance_indicators.append(has_proper_nouns)
    
    # 2. 检查是否要求讨论特定主题
    instruction_keywords = extract_keywords(instruction)
    doc_keywords = extract_keywords(doc)
    keyword_overlap = len(instruction_keywords & doc_keywords) / max(len(instruction_keywords), 1)
    compliance_indicators.append(keyword_overlap > 0.3)
    
    # 3. 检查是否包含具体信息（数字、日期等）
    if 'specific' in instruction_lower or 'detail' in instruction_lower:
        has_specifics = bool(re.search(r'\b\d+\b', doc) or re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', doc, re.I))
        compliance_indicators.append(has_specifics)
    
    # 返回综合判断
    return any(compliance_indicators) if compliance_indicators else True


def analyze_negative_quality(sample: Dict) -> Tuple[bool, str, List[Dict]]:
    """
    分析负面文档的质量
    返回：(是否缺乏 hard negatives, 原因, 详细分析)
    """
    query = sample.get('query', '')
    instruction = sample.get('instruction', '')
    negatives = sample.get('neg', [])
    
    analysis_results = []
    hard_negative_count = 0
    easy_negative_count = 0
    
    for i, neg_doc in enumerate(negatives):
        # 计算语义相似度
        similarity = calculate_semantic_similarity(query, neg_doc)
        
        # 检查是否符合 instruction
        complies = check_instruction_compliance(neg_doc, instruction)
        
        analysis = {
            'index': i,
            'similarity': similarity,
            'complies_with_instruction': complies,
            'doc_preview': neg_doc[:200] + '...' if len(neg_doc) > 200 else neg_doc
        }
        
        # Hard negative: 语义相关但不符合 instruction
        if similarity > 0.1 and not complies:
            analysis['type'] = 'hard_negative'
            hard_negative_count += 1
        # Easy negative: 语义不相关
        elif similarity < 0.05:
            analysis['type'] = 'easy_negative'
            easy_negative_count += 1
        # 其他情况
        else:
            analysis['type'] = 'medium_negative'
        
        analysis_results.append(analysis)
    
    # 判断标准：
    # 1. 如果所有负样本都是 easy negatives（相似度 < 0.05），则缺乏 hard negatives
    # 2. 如果没有 hard negatives，则认为缺乏
    lacks_hard_negatives = hard_negative_count == 0
    
    reason = f"Easy negatives: {easy_negative_count}/{len(negatives)}, Hard negatives: {hard_negative_count}/{len(negatives)}"
    
    return lacks_hard_negatives, reason, analysis_results


def main():
    input_file = '/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/train_data_igp.jsonl'
    output_file = '/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/lacks_hard_negatives.jsonl'
    
    print("=" * 80)
    print("训练数据集 Hard Negatives 质量分析")
    print("=" * 80)
    
    # 加载数据
    data = load_dataset(input_file)
    total_samples = len(data)
    print(f"\n总样本数: {total_samples}")
    
    # 分析每个样本
    problem_samples = []
    stats = {
        'lacks_hard_negatives': 0,
        'has_hard_negatives': 0,
        'total_negatives': 0,
        'easy_negatives': 0,
        'medium_negatives': 0,
        'hard_negatives': 0
    }
    
    for idx, sample in enumerate(data):
        lacks_hard, reason, analysis = analyze_negative_quality(sample)
        
        # 更新统计
        for a in analysis:
            stats['total_negatives'] += 1
            if a['type'] == 'easy_negative':
                stats['easy_negatives'] += 1
            elif a['type'] == 'hard_negative':
                stats['hard_negatives'] += 1
            else:
                stats['medium_negatives'] += 1
        
        if lacks_hard:
            stats['lacks_hard_negatives'] += 1
            problem_sample = {
                'original_index': idx,
                'query': sample['query'],
                'instruction': sample['instruction'],
                'pos': sample['pos'],
                'neg': sample['neg'],
                'reason': reason,
                'negative_analysis': analysis
            }
            problem_samples.append(problem_sample)
        else:
            stats['has_hard_negatives'] += 1
        
        # 每 100 个样本显示进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{total_samples} 样本...")
    
    # 保存问题样本
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in problem_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 打印统计报告
    print("\n" + "=" * 80)
    print("分析结果统计")
    print("=" * 80)
    print(f"\n总样本数: {total_samples}")
    print(f"缺乏 hard negatives 的样本: {stats['lacks_hard_negatives']} ({stats['lacks_hard_negatives']/total_samples*100:.1f}%)")
    print(f"包含 hard negatives 的样本: {stats['has_hard_negatives']} ({stats['has_hard_negatives']/total_samples*100:.1f}%)")
    print(f"\n负面文档类型分布:")
    print(f"  Easy negatives (语义无关): {stats['easy_negatives']} ({stats['easy_negatives']/stats['total_negatives']*100:.1f}%)")
    print(f"  Medium negatives (中等相关): {stats['medium_negatives']} ({stats['medium_negatives']/stats['total_negatives']*100:.1f}%)")
    print(f"  Hard negatives (相关但不符合): {stats['hard_negatives']} ({stats['hard_negatives']/stats['total_negatives']*100:.1f}%)")
    
    print(f"\n问题样本已保存到: {output_file}")
    print(f"共保存 {len(problem_samples)} 个样本")
    
    # 显示一些示例
    print("\n" + "=" * 80)
    print("缺乏 hard negatives 的样本示例（前 3 个）:")
    print("=" * 80)
    for i, sample in enumerate(problem_samples[:3]):
        print(f"\n示例 {i+1}:")
        print(f"Query: {sample['query'][:100]}...")
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"原因: {sample['reason']}")
        print(f"负面文档数量: {len(sample['neg'])}")


if __name__ == '__main__':
    main()
