import os
import time
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import heapq
import pickle

from cs336_basics.pretokenization_example import find_chunk_boundaries

tiny_stories_val_path = "data/TinyStoriesV2-GPT4-valid.txt"
tiny_stories_train_path = "data/TinyStoriesV2-GPT4-train.txt"
owt_train_path = "data/owt_train.txt"
owt_valid_path = "data/owt_valid.txt"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(
    file_path: str | os.PathLike, special_tokens: list[str], num_processes: int = 1
) -> dict[tuple[int], int]:
    start_time = time.time()

    if not file_path:
        raise ValueError("No file path passed to pre_tokenize.")

    pre_tokens_count = Counter()

    print(f"Starting pre-tokenization...")
    boundaries = None
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

    print(f"Pre-tokenization func, time taken: {time.time() - start_time}")
    return pre_tokens_count


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


def count_pairs(
    inp: dict[tuple[int, ...], int],
) -> tuple[
    dict[tuple[int, int], int], dict[tuple[int, int], set], dict[tuple[int, ...], set]
]:
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


def merge_pair(
    byte_tokens_count: dict[tuple[int, ...], int],
    pairs_to_sequences: dict[tuple[int, int], set],
    old: tuple[int, int],
    new: int,
):
    keys_to_remove = []
    keys_to_add = []
    for old_w in pairs_to_sequences[old]:
        i = 0
        w = list(old_w)
        while i < len(w) - 1:
            if w[i] == old[0] and w[i + 1] == old[1]:
                w[i] = new
                w.pop(i + 1)
            i += 1

        if len(w) != len(old_w):
            keys_to_remove.append(old_w)
            keys_to_add.append((w, byte_tokens_count[old_w]))

    return keys_to_remove, keys_to_add


def train_byte_pair_encoder(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not input_path or not vocab_size or not special_tokens:
        raise ValueError

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # how many merges we are allowed to make
    num_merges = vocab_size - len(vocab)

    # num_processes is 2 so we break the whole training set into 2 chunks
    byte_tokens_count = pre_tokenize(input_path, special_tokens, 2)
    count_start_time = time.time()
    pairs, pairs_to_sequences, seqs_to_pairs = count_pairs(byte_tokens_count)
    print(f"Time taken for building initial counts: {time.time() - count_start_time}")
    start_time = time.time()
    for _ in range(num_merges):
        if len(pairs) == 0:
            break

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

        # get back only new sequences and their counts
        keys_to_remove, keys_to_add = merge_pair(
            byte_tokens_count, pairs_to_sequences, max_pair, new_tok_idx
        )

        # remove the counts of all pairs from these sequences since we will recount them
        for s in keys_to_remove:
            for p in zip(s, s[1:]):
                pairs[p] -= byte_tokens_count[s]
                if pairs[p] == 0:
                    pairs.pop(p)

            for p in seqs_to_pairs[s]:
                if p in pairs_to_sequences:
                    pairs_to_sequences[p].remove(s)
                    if not pairs_to_sequences[p]:
                        pairs_to_sequences.pop(p)

            # remove the outdated sequences from play
            byte_tokens_count.pop(s)
            seqs_to_pairs.pop(s)

        # add in the new sequences with their counts
        for k, count in keys_to_add:
            byte_tokens_count[tuple(k)] = byte_tokens_count.get(tuple(k), 0) + count

        # count pairs and pairs_to_sequences again
        for w, count in keys_to_add:
            w = tuple(w)
            for j in range(len(w) - 1):
                pair = w[j : j + 2]
                pairs[pair] += count
                pairs_to_sequences[pair].add(w)
                seqs_to_pairs[w].add(pair)

    print(f"Time taken for merges ({num_merges}): {time.time() - start_time}")
    return vocab, merges


if __name__ == "__main__":
    start_time = time.time()
    data_path = owt_train_path
    vocab, merges = train_byte_pair_encoder(data_path, 32000, ["<|endoftext|>"])
    
    end_time = time.time()
    print(f"Total time：{end_time - start_time:.2f} 秒")
    # Extract filename without path and extension
    import os.path
    filename_base = os.path.splitext(os.path.basename(data_path))[0]
    
    os.makedirs("tokenizer_v2_results", exist_ok=True)
    with open(f"tokenizer_v2_results/{filename_base}_vocab.json", "wb") as f:
        f.write(pickle.dumps(vocab))
    with open(f"tokenizer_v2_results/{filename_base}_merges.json", "wb") as f:
        f.write(pickle.dumps(merges))