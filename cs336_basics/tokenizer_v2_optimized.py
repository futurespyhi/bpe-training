import os
import time
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import heapq
import pickle
import gc
import mmap
from typing import Iterator, Dict, Tuple, Set

from cs336_basics.pretokenization_example import find_chunk_boundaries

tiny_stories_val_path = "data/TinyStoriesV2-GPT4-valid.txt"
tiny_stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
owt_train_path = "data/owt_train.txt"
owt_valid_path = "data/owt_valid.txt"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 内存优化配置 - 针对30分钟限制优化
BATCH_SIZE = 80000  # 增大批次大小，减少IO次数
MAX_MEMORY_SEQUENCES = 200000  # 大幅增加内存容量，减少清理频率
CHECKPOINT_FREQUENCY = 1000  # 进一步减少检查点频率

# 速度优化配置
FAST_MODE_THRESHOLD = 3 * 1024 * 1024 * 1024  # 3GB以下文件使用快速模式
MEMORY_CLEANUP_THRESHOLD = 300000  # 更高的清理阈值，允许更多序列在内存中


def pre_tokenize_fast(
    file_path: str | os.PathLike, special_tokens: list[str], num_processes: int = 1
) -> dict[tuple[int], int]:
    """快速预分词，适用于小文件"""
    start_time = time.time()

    if not file_path:
        raise ValueError("No file path passed to pre_tokenize.")

    pre_tokens_count = Counter()
    print(f"Starting fast pre-tokenization...")
    
    split_expr = "(?:" + "|".join(re.escape(t) for t in special_tokens) + ")"

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    process_args = list(
        zip(
            boundaries,
            boundaries[1:],
            [file_path] * len(boundaries),
            [split_expr] * len(boundaries),
        )
    )
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, process_args)

    for r in results:
        pre_tokens_count.update(r)

    print(f"Fast pre-tokenization completed, time taken: {time.time() - start_time:.2f}s")
    return pre_tokens_count


def pre_tokenize_streaming(
    file_path: str | os.PathLike, 
    special_tokens: list[str], 
    num_processes: int = 1,
    batch_size: int = BATCH_SIZE
) -> Iterator[Dict[tuple[int], int]]:
    """流式预分词，分批返回token计数"""
    if not file_path:
        raise ValueError("No file path passed to pre_tokenize.")

    print(f"Starting streaming pre-tokenization with batch size {batch_size}...")
    boundaries = None
    split_expr = "(?:" + "|".join(re.escape(t) for t in special_tokens) + ")"

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 将边界分成更小的批次
    chunk_size = max(1, len(boundaries) // (num_processes * 4))  # 更小的chunk
    
    for i in range(0, len(boundaries) - 1, chunk_size):
        end_idx = min(i + chunk_size, len(boundaries) - 1)
        batch_boundaries = boundaries[i:end_idx + 1]
        
        process_args = list(
            zip(
                batch_boundaries[:-1],
                batch_boundaries[1:],
                [file_path] * (len(batch_boundaries) - 1),
                [split_expr] * (len(batch_boundaries) - 1),
            )
        )
        
        actual_processes = min(num_processes, len(process_args))
        print(f"  Using {actual_processes} processes for {len(process_args)} chunks")
        with Pool(processes=actual_processes) as pool:
            results = pool.starmap(process_chunk, process_args)

        batch_count = Counter()
        for r in results:
            batch_count.update(r)
        
        if batch_count:
            yield batch_count
        
        # 强制垃圾回收
        del results
        del batch_count
        gc.collect()


def process_chunk(start: int, end: int, file_path: str | os.PathLike, split_expr: str):
    tokens = []
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = re.split(split_expr, chunk)
        pattern = re.compile(PAT)
        for c in chunk:
            tokens.extend([match.encode("utf-8") for match in re.findall(pattern, c)])

    return Counter(tokens)


def count_pairs_batch(
    inp: dict[tuple[int, ...], int],
) -> tuple[
    dict[tuple[int, int], int], dict[tuple[int, int], set], dict[tuple[int, ...], set]
]:
    """批量计算pairs，优化内存使用"""
    count = defaultdict(int)
    pairs_to_sequences: dict[tuple[int, int], set] = defaultdict(set)
    sequences_to_pairs: dict[tuple[int, ...], set] = defaultdict(set)

    for w in inp:
        if len(w) <= 1:
            continue
        for k in zip(w, w[1:]):
            count[k] = count.get(k, 0) + inp[w]
            pairs_to_sequences[k].add(w)
            sequences_to_pairs[w].add(k)

    return count, pairs_to_sequences, sequences_to_pairs


def merge_pair_optimized(
    byte_tokens_count: dict[tuple[int, ...], int],
    pairs_to_sequences: dict[tuple[int, int], set],
    old: tuple[int, int],
    new: int,
):
    """优化的merge操作，减少内存分配"""
    keys_to_remove = []
    keys_to_add = []
    
    # 使用集合的副本避免修改时的问题
    sequences_to_process = list(pairs_to_sequences[old])
    
    for old_w in sequences_to_process:
        i = 0
        w = list(old_w)
        merged = False
        
        while i < len(w) - 1:
            if w[i] == old[0] and w[i + 1] == old[1]:
                w[i] = new
                w.pop(i + 1)
                merged = True
            else:
                i += 1

        if merged:
            keys_to_remove.append(old_w)
            keys_to_add.append((tuple(w), byte_tokens_count[old_w]))

    return keys_to_remove, keys_to_add


def save_checkpoint(vocab, merges, merge_count, checkpoint_dir="checkpoints"):
    """保存训练检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'vocab': vocab,
        'merges': merges,
        'merge_count': merge_count
    }
    
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{merge_count}.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved at merge {merge_count}")


def load_checkpoint(checkpoint_path):
    """加载训练检查点"""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def train_byte_pair_encoder_optimized(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str],
    checkpoint_dir: str = "checkpoints",
    resume_from: str = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 自动检测文件大小，使用快速模式
    file_size = os.path.getsize(input_path)
    is_small_file = file_size < FAST_MODE_THRESHOLD  # 3GB以下使用快速模式
    
    if is_small_file:
        return train_byte_pair_encoder_fast(input_path, vocab_size, special_tokens)
    else:
        return train_byte_pair_encoder_memory_optimized(input_path, vocab_size, special_tokens, checkpoint_dir, resume_from)


def train_byte_pair_encoder_fast(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """快速模式：适用于小文件，优先速度"""
    if not input_path or not vocab_size or not special_tokens:
        raise ValueError

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    
    file_size = os.path.getsize(input_path)
    print(f"Using FAST MODE for {file_size/(1024**3):.2f}GB file")
    
    # 快速预分词 - 使用更多进程
    byte_tokens_count = pre_tokenize_fast(input_path, special_tokens, min(cpu_count(), 8))
    
    # 一次性计算所有pairs
    pairs, pairs_to_sequences, seqs_to_pairs = count_pairs_batch(byte_tokens_count)
    
    print(f"Starting {num_merges} merges with {len(byte_tokens_count)} unique sequences")
    
    # 优化的merge循环
    merge_count = 0
    for _ in range(num_merges):
        if len(pairs) == 0:
            break

        # 直接找最大值，不使用heapq（对小数据更快）
        max_pair = max(pairs.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))
        max_pair_key = max_pair[0]
        max_pair_in_bytes = (vocab[max_pair_key[0]], vocab[max_pair_key[1]])

        merges.append(max_pair_in_bytes)
        new_tok_idx = len(vocab)
        vocab[new_tok_idx] = vocab[max_pair_key[0]] + vocab[max_pair_key[1]]

        # 快速更新数据结构
        keys_to_remove, keys_to_add = merge_pair_fast(
            byte_tokens_count, pairs_to_sequences, seqs_to_pairs, 
            pairs, max_pair_key, new_tok_idx
        )

        # 更新计数
        for s in keys_to_remove:
            byte_tokens_count.pop(s)

        for k, count in keys_to_add:
            byte_tokens_count[k] = count
            
        merge_count += 1
        if merge_count % 1000 == 0:
            print(f"Completed {merge_count}/{num_merges} merges ({merge_count/num_merges*100:.1f}%)")
            
        # 内存管理 - 仅在必要时清理
        if len(byte_tokens_count) > MEMORY_CLEANUP_THRESHOLD:
            print(f"Cleaning up sequences: {len(byte_tokens_count)} -> {MEMORY_CLEANUP_THRESHOLD//2}")
            sorted_items = sorted(byte_tokens_count.items(), key=lambda x: x[1], reverse=True)
            byte_tokens_count = dict(sorted_items[:MEMORY_CLEANUP_THRESHOLD//2])
            # 重新计算pairs
            pairs, pairs_to_sequences, seqs_to_pairs = count_pairs_batch(byte_tokens_count)
            gc.collect()

    return vocab, merges


def merge_pair_fast(
    byte_tokens_count: dict[tuple[int, ...], int],
    pairs_to_sequences: dict[tuple[int, int], set],
    sequences_to_pairs: dict[tuple[int, ...], set],
    pairs: dict[tuple[int, int], int],
    old: tuple[int, int],
    new: int,
):
    """快速merge操作，直接更新所有相关数据结构"""
    keys_to_remove = []
    keys_to_add = []
    
    sequences_to_process = list(pairs_to_sequences[old])
    
    for old_w in sequences_to_process:
        # 执行merge
        new_w = []
        i = 0
        while i < len(old_w):
            if i < len(old_w) - 1 and old_w[i] == old[0] and old_w[i + 1] == old[1]:
                new_w.append(new)
                i += 2
            else:
                new_w.append(old_w[i])
                i += 1
        
        new_w = tuple(new_w)
        if new_w != old_w:
            keys_to_remove.append(old_w)
            keys_to_add.append((new_w, byte_tokens_count[old_w]))
            
            # 立即更新pairs计数
            for p in zip(old_w, old_w[1:]):
                pairs[p] -= byte_tokens_count[old_w]
                if pairs[p] <= 0:
                    pairs.pop(p, None)
            
            # 清理pairs_to_sequences
            for p in sequences_to_pairs[old_w]:
                if p in pairs_to_sequences:
                    pairs_to_sequences[p].discard(old_w)
                    if not pairs_to_sequences[p]:
                        pairs_to_sequences.pop(p, None)
            
            sequences_to_pairs.pop(old_w, None)
            
            # 添加新序列的pairs
            for j in range(len(new_w) - 1):
                pair = (new_w[j], new_w[j + 1])
                if pair not in pairs:
                    pairs[pair] = 0
                    pairs_to_sequences[pair] = set()
                pairs[pair] += byte_tokens_count[old_w]
                pairs_to_sequences[pair].add(new_w)
                if new_w not in sequences_to_pairs:
                    sequences_to_pairs[new_w] = set()
                sequences_to_pairs[new_w].add(pair)

    return keys_to_remove, keys_to_add


def train_byte_pair_encoder_memory_optimized(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str],
    checkpoint_dir: str = "checkpoints",
    resume_from: str = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """内存优化模式：适用于大文件"""
    
    if not input_path or not vocab_size or not special_tokens:
        raise ValueError

    start_time = time.time()
    
    # 检查是否要从检查点恢复
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint_data = load_checkpoint(resume_from)
        vocab = checkpoint_data['vocab']
        merges = checkpoint_data['merges']
        merge_count_start = checkpoint_data['merge_count']
    else:
        # 初始化
        merges: list[tuple[bytes, bytes]] = []
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for st in special_tokens:
            vocab[len(vocab)] = st.encode("utf-8")
        merge_count_start = 0

    num_merges = vocab_size - len(vocab)
    
    print(f"Starting memory-optimized training with {num_merges} merges to perform...")
    
    # 流式处理数据，分批构建初始token计数
    print("Building initial token counts in batches...")
    byte_tokens_count = {}
    batch_count = 0
    
    for batch_tokens in pre_tokenize_streaming(input_path, special_tokens, 4, BATCH_SIZE):
        batch_count += 1
        print(f"Processing batch {batch_count}, current vocabulary size: {len(vocab)}")
        
        # 合并到主计数器
        for token, count in batch_tokens.items():
            if token in byte_tokens_count:
                byte_tokens_count[token] += count
            else:
                byte_tokens_count[token] = count
        
        # 内存管理：如果序列过多，清理低频序列
        if len(byte_tokens_count) > MAX_MEMORY_SEQUENCES:
            print(f"Memory limit reached, pruning low-frequency sequences: {len(byte_tokens_count)} -> {MAX_MEMORY_SEQUENCES//2}")
            # 保留高频序列
            sorted_items = sorted(byte_tokens_count.items(), key=lambda x: x[1], reverse=True)
            byte_tokens_count = dict(sorted_items[:MAX_MEMORY_SEQUENCES//2])
            gc.collect()
        
        del batch_tokens
        gc.collect()

    print(f"Initial token count completed. Total unique sequences: {len(byte_tokens_count)}")
    
    # 开始merge循环
    for merge_idx in range(merge_count_start, num_merges):
        print(f"Merge {merge_idx + 1}/{num_merges}")
        
        # 重新计算pairs（每次merge后重算，节省内存）
        pairs, pairs_to_sequences, seqs_to_pairs = count_pairs_batch(byte_tokens_count)
        
        if len(pairs) == 0:
            print("No more pairs to merge")
            break

        # 找到最频繁的pair
        sorted_pairs = heapq.nlargest(100, pairs.items(), key=lambda item: item[1])
        max_count = sorted_pairs[0][1]
        
        ties = []
        for pair, count in sorted_pairs:
            if count == max_count:
                ties.append((vocab[pair[0]], vocab[pair[1]]))
            else:
                break

        max_index, _ = max(enumerate(ties), key=lambda x: x[1])
        max_pair = sorted_pairs[max_index][0]
        max_pair_in_bytes = (vocab[max_pair[0]], vocab[max_pair[1]])

        merges.append(max_pair_in_bytes)
        new_tok_idx = len(vocab)
        vocab[len(vocab)] = vocab[max_pair[0]] + vocab[max_pair[1]]

        print(f"Merging pair {max_pair_in_bytes} with count {max_count}, new token idx: {new_tok_idx}")

        # 执行merge
        keys_to_remove, keys_to_add = merge_pair_optimized(
            byte_tokens_count, pairs_to_sequences, max_pair, new_tok_idx
        )

        # 更新数据结构
        for s in keys_to_remove:
            byte_tokens_count.pop(s, None)

        for k, count in keys_to_add:
            byte_tokens_count[k] = byte_tokens_count.get(k, 0) + count

        # 清理内存
        del pairs, pairs_to_sequences, seqs_to_pairs
        gc.collect()

        # 定期保存检查点
        if (merge_idx + 1) % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(vocab, merges, merge_idx + 1, checkpoint_dir)
        
        # 内存管理
        if len(byte_tokens_count) > MAX_MEMORY_SEQUENCES:
            print("Pruning low-frequency sequences...")
            sorted_items = sorted(byte_tokens_count.items(), key=lambda x: x[1], reverse=True)
            byte_tokens_count = dict(sorted_items[:MAX_MEMORY_SEQUENCES//2])
            gc.collect()

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # 保存最终检查点
    save_checkpoint(vocab, merges, len(merges), checkpoint_dir)
    
    return vocab, merges


if __name__ == "__main__":
    start_time = time.time()
    data_path = owt_train_path
    
    # 可以从检查点恢复训练
    # resume_checkpoint = "checkpoints/checkpoint_1000.pkl"  # 如果需要恢复
    resume_checkpoint = None
    
    vocab, merges = train_byte_pair_encoder_optimized(
        data_path, 
        32000, 
        ["<|endoftext|>"],
        resume_from=resume_checkpoint
    )
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # 保存结果
    import os.path
    filename_base = os.path.splitext(os.path.basename(data_path))[0]
    
    os.makedirs("tokenizer_v2_results", exist_ok=True)
    with open(f"tokenizer_v2_results/{filename_base}_vocab_optimized.json", "wb") as f:
        f.write(pickle.dumps(vocab))
    with open(f"tokenizer_v2_results/{filename_base}_merges_optimized.json", "wb") as f:
        f.write(pickle.dumps(merges))
    
    print(f"Results saved to tokenizer_v2_results/")