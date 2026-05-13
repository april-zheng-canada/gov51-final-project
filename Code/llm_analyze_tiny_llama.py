import sys
import os
import csv
import json
import argparse
import logging
from datetime import datetime

# Ensure user site-packages is on path
USER_SITE = os.path.join(os.environ.get("APPDATA", ""), "Python", "Python311", "site-packages")
if os.path.isdir(USER_SITE) and USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_analyze.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

INPUT_DIR = os.path.join(os.path.dirname(__file__), "filtered_science")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_results")

COMMENT_FILES = [
    "moderatepolitics_comments.csv",
    "science_comments.csv",
]

# CSV columns (no header): score, date, author, permalink, body
COL_SCORE = 0
COL_DATE = 1
COL_AUTHOR = 2
COL_PERMALINK = 3
COL_BODY = 4

SYSTEM_PROMPT = (
    "You are a research assistant that classifies Reddit comments. "
    "For each comment, output ONLY a JSON object with exactly two keys:\n"
    '  "trust": one of "trusting" or "non-trusting"\n'
    '  "leaning": one of "liberal", "conservative", or "neutral"\n'
    "Definitions (important):\n"
    '- "trusting": explicit confidence, support, approval, or positive acceptance of science, evidence, experts, institutions, or findings.\n'
    '- "non-trusting": explicit distrust, rejection, conspiracy framing, accusations of deception, or strong skepticism toward science, evidence, experts, institutions, or findings.\n'
    "Decision rules for trust:\n"
    '- If the comment is merely descriptive, off-topic, joking, or ambiguous, label "trusting" unless there is clear distrust language.\n'
    '- Do NOT label "non-trusting" just because the tone is critical, sarcastic, uncertain, or asks questions.\n'
    '- Use "non-trusting" only when distrust/rejection is clearly stated or strongly implied.\n'
    "Quick examples:\n"
    '- "Great study, solid evidence" -> trusting\n'
    '- "Scientists are lying to us" -> non-trusting\n'
    '- "Interesting article, thanks for sharing" -> trusting\n'
    '- "I am not sure yet" -> trusting\n'
    '- "liberal": the author expresses progressive, left-leaning views.\n'
    '- "conservative": the author expresses traditionalist, right-leaning views.\n'
    '- "neutral": the author does not clearly lean liberal or conservative.\n'
    "Output ONLY the JSON object, no other text."
)


def build_prompt(comment_text: str, tokenizer) -> str:
    """Build a prompt for an instruction-tuned Llama-style model."""
    truncated = comment_text[:600]
    user_prompt = (
        "Classify the following Reddit comment and return only JSON with keys "
        '"trust" and "leaning".\n\n'
        f"Reddit comment:\n{truncated}"
    )

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nClassification:"


def parse_model_output(raw_output: str) -> dict:
    """Extract trust and leaning from model output, with fallbacks."""
    # Try to find JSON in the output
    text = raw_output.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        return normalize_result(obj)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end])
            return normalize_result(obj)
        except json.JSONDecodeError:
            pass

    # Fallback: keyword matching
    text_lower = text.lower()
    trust = "trusting" if "trusting" in text_lower and "non" not in text_lower.split("trusting")[0][-4:] else "non-trusting"
    if "liberal" in text_lower:
        leaning = "liberal"
    elif "conservative" in text_lower:
        leaning = "conservative"
    else:
        leaning = "neutral"

    return {"trust": trust, "leaning": leaning}


def normalize_result(obj: dict) -> dict:
    """Normalize parsed JSON to expected values."""
    trust = str(obj.get("trust", "")).lower().strip()
    leaning = str(obj.get("leaning", "")).lower().strip()

    if "non" in trust:
        trust = "non-trusting"
    elif "trust" in trust:
        trust = "trusting"
    else:
        trust = "non-trusting"

    if leaning not in ("liberal", "conservative", "neutral"):
        leaning = "neutral"

    return {"trust": trust, "leaning": leaning}


def load_model(model_name: str):
    """Load the selected causal LM and tokenizer with GPU support."""
    log.info(f"Loading model: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Some causal LM tokenizers do not define a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    log.info(f"Model loaded on {device}")
    return model, tokenizer, device


def classify_comment(model, tokenizer, device: str, comment_text: str) -> dict:
    """Run inference on a single comment and return classification."""
    prompt = build_prompt(comment_text, tokenizer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 2048), 2048),
        padding=False,
    )
    # Move inputs to correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_model_output(raw_output)


def get_processed_ids(output_file: str) -> set:
    """Load already-processed permalink IDs from output file for resume support."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row.get("permalink", ""))
    return processed


def count_csv_rows(file_path: str) -> int:
    """Count total rows in a CSV file (no header expected in input)."""
    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for _ in reader:
            total += 1
    return total


def process_file(model, tokenizer, device, input_path, output_path, limit, skip_existing, progress_step, classify_step, print_step):
    """Process a single comments CSV file."""
    log.info(f"Processing: {input_path}")
    progress_step = max(1, int(progress_step))
    classify_step = max(1, int(classify_step))
    print_step = max(1, int(print_step))

    processed_ids = set()
    file_exists = os.path.exists(output_path)
    if skip_existing and file_exists:
        processed_ids = get_processed_ids(output_path)
        log.info(f"Resuming: {len(processed_ids)} already processed")

    total_rows = count_csv_rows(input_path)
    log.info(f"Input rows: {total_rows}")

    mode = "a" if file_exists and skip_existing else "w"
    out_handle = open(output_path, mode, encoding="utf-8", newline="")
    writer = csv.writer(out_handle)

    if mode == "w":
        writer.writerow(["score", "date", "author", "permalink", "body_preview", "trust", "leaning"])

    count = 0
    skipped = 0
    scanned = 0
    eligible = 0
    sampled_out = 0
    trusting_count = 0
    non_trusting_count = 0
    liberal_count = 0
    conservative_count = 0
    neutral_count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            scanned += 1
            if len(row) < 5:
                continue

            permalink = row[COL_PERMALINK]
            if permalink in processed_ids:
                skipped += 1
                continue

            body = row[COL_BODY]
            if not body or len(body.strip()) < 20:
                continue

            eligible += 1
            # Classify only one eligible comment out of each classify_step comments.
            if eligible % classify_step != 0:
                sampled_out += 1
                continue

            result = classify_comment(model, tokenizer, device, body)

            body_preview = body[:200].replace("\n", " ")
            writer.writerow([
                row[COL_SCORE],
                row[COL_DATE],
                row[COL_AUTHOR],
                permalink,
                body_preview,
                result["trust"],
                result["leaning"],
            ])

            count += 1
            if result["trust"] == "trusting":
                trusting_count += 1
            else:
                non_trusting_count += 1

            if result["leaning"] == "liberal":
                liberal_count += 1
            elif result["leaning"] == "conservative":
                conservative_count += 1
            else:
                neutral_count += 1

            if count % print_step == 0:
                trust_pct = (trusting_count / count * 100.0) if count else 0.0
                non_trust_pct = (non_trusting_count / count * 100.0) if count else 0.0
                lib_pct = (liberal_count / count * 100.0) if count else 0.0
                cons_pct = (conservative_count / count * 100.0) if count else 0.0
                neu_pct = (neutral_count / count * 100.0) if count else 0.0
                log.info(
                    "  Summary[%d]: trust(trusting=%d %.2f%%, non-trusting=%d %.2f%%), "
                    "leaning(liberal=%d %.2f%%, conservative=%d %.2f%%, neutral=%d %.2f%%)",
                    count,
                    trusting_count,
                    trust_pct,
                    non_trusting_count,
                    non_trust_pct,
                    liberal_count,
                    lib_pct,
                    conservative_count,
                    cons_pct,
                    neutral_count,
                    neu_pct,
                )

            if count % progress_step == 0:
                out_handle.flush()
                pct = (scanned / total_rows * 100.0) if total_rows else 0.0
                log.info(
                    f"  Progress: processed={count}, skipped={skipped}, sampled_out={sampled_out}, "
                    f"scanned={scanned}/{total_rows} ({pct:.2f}%)"
                )

            if limit and count >= limit:
                log.info(f"  Reached limit of {limit}")
                break

    out_handle.close()
    log.info(f"Done: {count} comments classified, saved to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Classify Reddit comments using a Llama-family instruction model for trust and political leaning."
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max comments to process per file (default: 100, use 0 for all)"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from previous run, skipping already-processed comments"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME}). Use a Llama-family instruction-tuned causal LM for best results."
    )
    parser.add_argument(
        "--files", nargs="+", default=COMMENT_FILES,
        help="Comment CSV files to process (in filtered_science/)"
    )
    parser.add_argument(
        "--progress-step", type=int, default=10,
        help="Log progress every N processed comments (default: 10)"
    )
    parser.add_argument(
        "--classify-step", type=int, default=10,
        help="Run classification once every N eligible comments (default: 10)"
    )
    parser.add_argument(
        "--print-step", type=int, default=1000,
        help="Print running summary stats every N processed comments (default: 1000)"
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer, device = load_model(args.model)

    total = 0
    for filename in args.files:
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path):
            log.warning(f"File not found, skipping: {input_path}")
            continue

        output_name = filename.replace(".csv", "_analyzed.csv")
        output_path = os.path.join(OUTPUT_DIR, output_name)

        count = process_file(
            model, tokenizer, device,
            input_path, output_path, limit,
            args.resume, args.progress_step, args.classify_step, args.print_step
        )
        total += count

    log.info(f"All done. Total comments classified: {total}")


if __name__ == "__main__":
    main()
