import os
import time
import argparse
from typing import Literal
from multiprocessing import Pool

import regex as re
import numpy as np
import h5py

# add packages for optimization
import gc
import psutil

from pretokenization_example import find_chunk_boundaries
from tokenizer import Tokenizer


def get_tokenizer(dataset: Literal["ts", "owt"]) -> Tokenizer:
    if dataset == "ts":
        return Tokenizer.from_files(
            "tokenizer_v2_results/TinyStoriesV2-GPT4-train_vocab.json",
            "tokenizer_v2_results/TinyStoriesV2-GPT4-train_merges.json",
            ["<|endoftext|>"],
        )
    if dataset == "owt":
        return Tokenizer.from_files(
            "tokenizer_v2_results/owt_train_vocab_optimized.json",
            "tokenizer_v2_results/owt_train_merges_optimized.json",
            ["<|endoftext|>"],
        )


def get_docs(dataset: Literal["ts", "owt"], num_samples: int = 10) -> list[str]:
    ts_path = "data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "data/owt_train.txt"
    data_path = ts_path if dataset == "ts" else owt_path

    with open(data_path, "r") as f:
        doc_count = 0
        tmp = []
        while doc_count < num_samples - 1:
            tmp.extend(f.readlines(100))
            eot_tokens = re.findall(r"<\|endoftext\|>", "".join(tmp))
            doc_count = len(eot_tokens)

        docs = re.split(r"<\|endoftext\|>", "".join(tmp))
    return docs[:10]


def question_a():
    ts_samples = get_docs("ts")
    owt_samples = get_docs("owt")

    ts_tokenizer = get_tokenizer("ts")
    owt_tokenizer = get_tokenizer("owt")

    ts_encoded = [ts_tokenizer.encode(s) for s in ts_samples]
    owt_encoded = [owt_tokenizer.encode(s) for s in owt_samples]

    len_bytes_ts = [len(b.encode("utf-8")) for b in ts_samples]
    len_tokens_ts = [len(tkns) for tkns in ts_encoded]
    print(f"Bytes in TinyStories: {sum(len_bytes_ts)}")
    print(f"Tokens in TinyStories: {sum(len_tokens_ts)}")
    print(f"Compression Ratio: {sum(len_bytes_ts) / sum(len_tokens_ts)}")

    len_bytes_owt = [len(b.encode("utf-8")) for b in owt_samples]
    len_tokens_owt = [len(tkns) for tkns in owt_encoded]
    print(f"Bytes in OWT: {sum(len_bytes_owt)}")
    print(f"Tokens in OWT: {sum(len_tokens_owt)}")
    print(f"Compression Ratio: {sum(len_bytes_owt) / sum(len_tokens_owt)}")


def question_b():
    owt_samples = get_docs("owt")

    ts_tokenizer = get_tokenizer("ts")

    owt_encoded = [ts_tokenizer.encode(s) for s in owt_samples]

    len_bytes_owt = [len(b.encode("utf-8")) for b in owt_samples]
    len_tokens_owt = [len(tkns) for tkns in owt_encoded]
    print(f"Bytes in OWT: {sum(len_bytes_owt)}")
    print(f"Tokens in OWT (encoded with TS tokenizer): {sum(len_tokens_owt)}")
    print(
        f"Compression Ratio (encoded with TS tokenizer): {sum(len_bytes_owt) / sum(len_tokens_owt)}"
    )


def question_c():
   owt_samples = get_docs("owt")
   ts_tokenizer = get_tokenizer("ts")
   owt_tokenizer = get_tokenizer("owt")
   num_bytes_in_pile = 825 * 1024 * 1024 * 1024

   # Test cross-domain performance (original design)
   print("=== Cross-domain (OWT data + TS tokenizer) ===")
   start_time = time.time()
   _ = [ts_tokenizer.encode(s) for s in owt_samples]
   end_time = time.time()

   len_bytes_owt = [len(b.encode("utf-8")) for b in owt_samples]
   time_taken_s = end_time - start_time
   throughput_cross = sum(len_bytes_owt) / time_taken_s
   print(f"Bytes in OWT: {sum(len_bytes_owt)}")
   print(f"Throughput (bytes/second): {throughput_cross}")
   print(f"Estimated time to tokenize Pile (825 GB): {num_bytes_in_pile / throughput_cross:.2f} seconds")
   
   print("\n" + "="*50)
   
   # Test matched-domain performance (your modification)
   print("=== Matched-domain (OWT data + OWT tokenizer) ===")
   start_time = time.time()
   _ = [owt_tokenizer.encode(s) for s in owt_samples]
   end_time = time.time()

   time_taken_s = end_time - start_time
   throughput_matched = sum(len_bytes_owt) / time_taken_s
   print(f"Bytes in OWT: {sum(len_bytes_owt)}")
   print(f"Throughput (bytes/second): {throughput_matched}")
   print(f"Estimated time to tokenize Pile (825 GB): {num_bytes_in_pile / throughput_matched:.2f} seconds")
   
   print(f"\nPerformance difference: {throughput_matched/throughput_cross:.2f}x faster with matched tokenizer")


def question_d_debug(dataset="ts_valid"):
    """Debug version with single process, detailed logging, and smaller chunks"""
    # Configuration for debugging
    CHUNK_SIZE_MB = 1

    datasets = {
        "ts_valid": {"path": "data/TinyStoriesV2-GPT4-valid.txt", "tokenizer": get_tokenizer("ts")},
        "owt_valid": {"path": "data/owt_valid.txt", "tokenizer": get_tokenizer("owt")},
        "ts_train": {"path": "data/TinyStoriesV2-GPT4-train.txt", "tokenizer": get_tokenizer("ts")},
        "owt_train": {"path": "data/owt_train.txt", "tokenizer": get_tokenizer("owt")}
    }

    if dataset not in datasets:
        print(f"Invalid dataset: {dataset}. Choose from: {list(datasets.keys())}")
        return

    config = datasets[dataset]
    file_path = config["path"]
    tokenizer = config["tokenizer"]

    print(f"[DEBUG] Processing {dataset}: {file_path}")
    total_bytes = os.path.getsize(file_path)
    print(f"[DEBUG] File size: {total_bytes / (1024*1024*1024):.2f} GB")

    # Get chunk boundaries
    split_expr = "<|endoftext|>".encode("utf-8")
    with open(file_path, "rb") as f:
        desired_chunks = total_bytes // (CHUNK_SIZE_MB * 1024 * 1024)
        print(f"[DEBUG] Target number of chunks: {desired_chunks}")
        boundaries = find_chunk_boundaries(f, desired_chunks, split_expr)

    print(f"[DEBUG] Created {len(boundaries)-1} chunks")

    # Process single-threaded for debugging
    final_path = file_path.split(".txt")[0] + "_dataset_debug.h5"
    total_tokens = _process_chunks_single_thread(boundaries, file_path, tokenizer, final_path)

    print(f"[DEBUG] Completed processing {dataset}, total tokens: {total_tokens}")

def _process_chunks_single_thread(boundaries, file_path, tokenizer, output_path):
    """Single-threaded processing with detailed logging"""
    print(f"[DEBUG] Starting single-threaded processing -> {output_path}")

    with h5py.File(output_path, "w") as hf:
        # Create dataset with initial size
        ds = hf.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.uint16,
                             chunks=(10000,), compression='gzip', compression_opts=1)

        total_tokens = 0
        with open(file_path, "rb") as f:
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                print(f"[DEBUG] Processing chunk {i+1}/{len(boundaries)-1}, bytes {start}-{end}")

                # Read chunk
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_size_mb = len(chunk_bytes) / (1024 * 1024)
                print(f"[DEBUG] Chunk size: {chunk_size_mb:.2f} MB")

                try:
                    chunk_text = chunk_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    chunk_text = chunk_bytes.decode("utf-8", errors="replace")
                    print(f"[DEBUG] Warning: Unicode decode errors in chunk {i+1}")

                print(f"[DEBUG] Text length: {len(chunk_text)} chars")
                print(f"[DEBUG] Starting tokenization...")

                # Time the tokenization
                import time
                start_time = time.time()
                tokens = tokenizer.encode(chunk_text)
                tokenize_time = time.time() - start_time

                print(f"[DEBUG] Tokenization completed in {tokenize_time:.2f}s")
                print(f"[DEBUG] Generated {len(tokens)} tokens")

                # Write tokens to H5 file immediately
                if tokens:
                    _write_tokens_to_dataset_debug(ds, tokens, total_tokens)
                    total_tokens += len(tokens)
                    print(f"[DEBUG] Written {len(tokens)} tokens, total: {total_tokens}")

                # Force flush every chunk for debugging
                hf.flush()
                print(f"[DEBUG] H5 file flushed")

                # Cleanup
                del chunk_bytes, chunk_text, tokens
                gc.collect()

                print(f"[DEBUG] Chunk {i+1} completed\n")

        return total_tokens

def _write_tokens_to_dataset_debug(dataset, tokens, current_size):
    """Debug version with forced flush"""
    if not tokens:
        return

    print(f"[DEBUG] Writing {len(tokens)} tokens at position {current_size}")
    tokens_array = np.array(tokens, dtype=np.uint16)
    new_size = current_size + len(tokens_array)
    dataset.resize((new_size,))
    dataset[current_size:new_size] = tokens_array

    # Force write to disk immediately
    dataset.file.flush()
    print(f"[DEBUG] Dataset flushed to disk")

    del tokens_array

def question_d_optimized(dataset="ts_valid"):
    """Optimized parallel version with better performance for large files"""
    import uuid
    from multiprocessing import Pool, cpu_count

    datasets = {
        "ts_valid": {"path": "data/TinyStoriesV2-GPT4-valid.txt", "tokenizer": get_tokenizer("ts")},
        "owt_valid": {"path": "data/owt_valid.txt", "tokenizer": get_tokenizer("owt")},
        "ts_train": {"path": "data/TinyStoriesV2-GPT4-train.txt", "tokenizer": get_tokenizer("ts")},
        "owt_train": {"path": "data/owt_train.txt", "tokenizer": get_tokenizer("owt")}
    }

    if dataset not in datasets:
        print(f"Invalid dataset: {dataset}. Choose from: {list(datasets.keys())}")
        return

    config = datasets[dataset]
    file_path = config["path"]
    tokenizer = config["tokenizer"]

    total_bytes = os.path.getsize(file_path)
    file_size_gb = total_bytes / (1024**3)

    # Dynamic configuration based on file size and dataset
    if dataset.startswith("owt") and file_size_gb > 5:
        # OWT tokenizer is slow, use aggressive parallelization
        CHUNK_SIZE_MB = 2  # Small chunks for progress visibility
        MAX_PROCESSES = min(32, cpu_count())  # Use many processes
    elif file_size_gb > 1:
        CHUNK_SIZE_MB = 5
        MAX_PROCESSES = min(16, cpu_count())
    else:
        CHUNK_SIZE_MB = 10
        MAX_PROCESSES = min(8, cpu_count())

    print(f"[OPT] Processing {dataset}: {file_path}")
    print(f"[OPT] File size: {file_size_gb:.2f} GB")
    print(f"[OPT] Configuration: {MAX_PROCESSES} processes, {CHUNK_SIZE_MB}MB chunks")

    # Get chunk boundaries
    split_expr = "<|endoftext|>".encode("utf-8")
    with open(file_path, "rb") as f:
        desired_chunks = total_bytes // (CHUNK_SIZE_MB * 1024 * 1024)
        boundaries = find_chunk_boundaries(f, desired_chunks, split_expr)

    total_chunks = len(boundaries) - 1
    print(f"[OPT] Created {total_chunks} chunks ({CHUNK_SIZE_MB}MB each)")

    # Estimate time
    if dataset.startswith("owt"):
        est_time_per_chunk = CHUNK_SIZE_MB * 60  # ~60s per MB for OWT
        est_total_hours = (total_chunks * est_time_per_chunk) / MAX_PROCESSES / 3600
        print(f"[OPT] Estimated time: {est_total_hours:.1f} hours")

    # Create temp directory for this run
    temp_dir = f"data/temp_tokens_{uuid.uuid4().hex[:8]}"
    os.makedirs(temp_dir, exist_ok=True)

    # Create chunk assignments for each process
    chunk_groups = _create_optimized_chunk_groups(
        boundaries, file_path, tokenizer, MAX_PROCESSES, temp_dir
    )

    print(f"[OPT] Starting {MAX_PROCESSES} parallel processes...")
    start_time = time.time()

    # Process chunks in parallel with progress tracking
    temp_files = []
    completed_chunks = 0

    with Pool(MAX_PROCESSES) as pool:
        # Use imap_unordered for progress updates
        for result in pool.imap_unordered(_process_chunk_group_optimized, chunk_groups):
            if result:
                temp_file, chunks_processed = result
                temp_files.append(temp_file)
                completed_chunks += chunks_processed

                # Progress update
                elapsed = time.time() - start_time
                rate = completed_chunks / elapsed
                eta = (total_chunks - completed_chunks) / rate if rate > 0 else 0
                print(f"[OPT] Progress: {completed_chunks}/{total_chunks} chunks "
                      f"({completed_chunks*100/total_chunks:.1f}%), "
                      f"Rate: {rate:.2f} chunks/s, ETA: {eta/3600:.1f}h")

    print(f"[OPT] All processing complete, merging {len(temp_files)} files...")

    # Merge temporary files into final dataset
    final_path = file_path.split(".txt")[0] + "_dataset.h5"
    total_tokens = _merge_temp_files_streaming(temp_files, final_path)

    # Cleanup
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    os.rmdir(temp_dir)

    elapsed_total = time.time() - start_time
    print(f"[OPT] Completed in {elapsed_total/3600:.2f} hours")
    print(f"[OPT] Total tokens: {total_tokens:,}")
    print(f"[OPT] Output file: {final_path}")


def _create_optimized_chunk_groups(boundaries, file_path, tokenizer, num_processes, temp_dir):
    """Create optimized chunk groups with better load balancing"""
    import uuid
    total_chunks = len(boundaries) - 1
    chunks_per_process = total_chunks // num_processes
    remainder = total_chunks % num_processes

    chunk_groups = []
    chunk_idx = 0

    for proc_id in range(num_processes):
        # Distribute remainder chunks among first processes
        process_chunks = chunks_per_process + (1 if proc_id < remainder else 0)

        # Create chunk list for this process
        chunks = []
        for _ in range(process_chunks):
            if chunk_idx < total_chunks:
                chunks.append((boundaries[chunk_idx], boundaries[chunk_idx + 1]))
                chunk_idx += 1

        temp_file = os.path.join(temp_dir, f"proc_{proc_id}_{uuid.uuid4().hex[:8]}.h5")

        chunk_groups.append({
            'proc_id': proc_id,
            'chunks': chunks,
            'file_path': file_path,
            'temp_file': temp_file,
            'tokenizer': tokenizer,
            'total_chunks': len(chunks)
        })

    return chunk_groups

def _process_chunk_group_optimized(group):
    """Optimized chunk processing with better error handling and progress"""
    proc_id = group['proc_id']
    chunks = group['chunks']
    file_path = group['file_path']
    temp_file = group['temp_file']
    tokenizer = group['tokenizer']
    total_chunks = group['total_chunks']

    print(f"[P{proc_id:02d}] Starting: {total_chunks} chunks -> {os.path.basename(temp_file)}")

    try:
        with h5py.File(temp_file, "w") as hf:
            # Smaller chunks for better memory management
            ds = hf.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.uint16,
                                 chunks=(10000,), compression='gzip', compression_opts=1)

            total_tokens = 0
            with open(file_path, "rb") as f:
                for idx, (start, end) in enumerate(chunks):
                    # Read and process chunk
                    f.seek(start)
                    chunk_bytes = f.read(end - start)

                    try:
                        chunk_text = chunk_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

                    # Tokenize with timing
                    import time
                    start_time = time.time()
                    tokens = tokenizer.encode(chunk_text)
                    tokenize_time = time.time() - start_time

                    # Write tokens
                    if tokens:
                        _write_tokens_to_dataset(ds, tokens, total_tokens)
                        total_tokens += len(tokens)

                    # Periodic progress and flush
                    if (idx + 1) % 10 == 0:
                        hf.flush()
                        print(f"[P{proc_id:02d}] Progress: {idx+1}/{total_chunks} chunks, "
                              f"{total_tokens:,} tokens, last chunk: {tokenize_time:.1f}s")

                    # Cleanup
                    del chunk_bytes, chunk_text, tokens
                    if (idx + 1) % 20 == 0:
                        gc.collect()

            # Final flush
            hf.flush()
            print(f"[P{proc_id:02d}] Complete: {total_tokens:,} tokens")
            return temp_file, total_chunks

    except Exception as e:
        print(f"[P{proc_id:02d}] Error: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None

def _merge_temp_files_streaming(temp_files, final_path):
    """Streaming merge to avoid loading all data at once - Fixed Version"""
    import re
    
    print(f"[MERGE] Starting merge of {len(temp_files)} files -> {final_path}")

    # Fix 1: Ensure correct numerical ordering of temp files
    def extract_proc_id(filename):
        match = re.search(r'proc_(\d+)_', filename)
        return int(match.group(1)) if match else 999

    # Sort files by process ID numerically, not alphabetically
    temp_files_sorted = sorted([f for f in temp_files if os.path.exists(f)], 
                              key=extract_proc_id)
    
    print(f"[MERGE] Processing {len(temp_files_sorted)} valid temp files")

    # Quick validation of file completeness
    print(f"[MERGE] Validating {len(temp_files_sorted)} files...")
    expected_proc_ids = set(range(32))  # Assuming 32 processes based on your temp files
    actual_proc_ids = set()
    for f in temp_files_sorted:
        match = re.search(r'proc_(\d+)_', f)
        if match:
            actual_proc_ids.add(int(match.group(1)))

    if len(actual_proc_ids) != len(expected_proc_ids):
        print(f"[MERGE] WARNING: Expected 32 files, found {len(actual_proc_ids)}")
        missing_ids = expected_proc_ids - actual_proc_ids
        if missing_ids:
            print(f"[MERGE] Missing process IDs: {sorted(missing_ids)}")

    with h5py.File(final_path, "w") as final_hf:
        # Larger chunks for final file
        final_ds = final_hf.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.uint16,
                                         chunks=(100000,), compression='gzip', compression_opts=1)

        total_tokens = 0
        failed_files = []  # Track files that fail to merge

        # Fix 2: Add comprehensive error handling for each file
        for i, temp_file in enumerate(temp_files_sorted):
            try:
                print(f"[MERGE] Processing file {i+1}/{len(temp_files_sorted)}: {os.path.basename(temp_file)}")

                with h5py.File(temp_file, "r") as temp_hf:
                    # Fix 3: Verify dataset exists before processing
                    if "data" not in temp_hf:
                        print(f"[MERGE] Warning: No 'data' dataset in {temp_file}, skipping")
                        continue
                    
                    temp_data = temp_hf["data"]
                    temp_size = len(temp_data)
                    
                    if temp_size == 0:
                        print(f"[MERGE] Warning: Empty dataset in {temp_file}, skipping")
                        continue

                    # Stream in chunks to avoid loading entire file
                    chunk_size = 1000000  # 1M tokens at a time
                    file_tokens_processed = 0

                    for start in range(0, temp_size, chunk_size):
                        end = min(start + chunk_size, temp_size)
                        chunk = temp_data[start:end]

                        if len(chunk) > 0:
                            # Fix 4: Ensure robust dataset writing
                            old_size = final_ds.shape[0]
                            new_size = old_size + len(chunk)
                            final_ds.resize((new_size,))
                            final_ds[old_size:new_size] = chunk
                            total_tokens += len(chunk)
                            file_tokens_processed += len(chunk)

                    print(f"[MERGE] Added {file_tokens_processed:,} tokens from {os.path.basename(temp_file)}, total: {total_tokens:,}")

            except Exception as e:
                print(f"[MERGE] ERROR processing {temp_file}: {e}")
                failed_files.append(temp_file)  # Record failed file
                print(f"[MERGE] Continuing with remaining files...")
                continue

            # Periodic flush for stability
            if (i + 1) % 5 == 0:
                final_hf.flush()
                print(f"[MERGE] Flushed data after {i+1} files")

        # Final flush
        final_hf.flush()

    # Report any failed files
    if failed_files:
        print(f"[MERGE] WARNING: {len(failed_files)} files failed to merge")
        print(f"[MERGE] Failed files: {[os.path.basename(f) for f in failed_files]}")

    print(f"[MERGE] Complete: {total_tokens:,} total tokens written to {final_path}")
    print(f"[MERGE] Successfully merged {len(temp_files_sorted) - len(failed_files)}/{len(temp_files_sorted)} files")
    return total_tokens


def _write_tokens_to_dataset(dataset, tokens, current_size):
    """Write tokens to H5 dataset"""
    if not tokens:
        return
    
    tokens_array = np.array(tokens, dtype=np.uint16)
    new_size = current_size + len(tokens_array)
    dataset.resize((new_size,))
    dataset[current_size:new_size] = tokens_array
    del tokens_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("q", type=str, help="Question to run (a/b/c/d_debug/d_opt)")
    parser.add_argument("dataset", nargs="?", default="ts_valid", type=str,
                       help="Dataset for question d (ts_valid/owt_valid/ts_train/owt_train)")

    args = parser.parse_args()
    if args.q == "a":
        question_a()
    elif args.q == "b":
        question_b()
    elif args.q == "c":
        question_c()
    elif args.q == "d_debug":
        question_d_debug(args.dataset)
    elif args.q == "d_opt":
        question_d_optimized(args.dataset)
    else:
        print(f"Invalid question: {args.q}. Choose from: a, b, c, d_debug, d_opt")