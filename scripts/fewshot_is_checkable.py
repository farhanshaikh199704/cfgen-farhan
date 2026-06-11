"""
Few-shot is_checkable evaluation experiment.

Iterates over different few-shot configurations using consensus calibration
examples, evaluates on test set, and reports agreement metrics.
"""

import pandas as pd
import numpy as np
import re
import time
import logging
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import openai

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Prompts ---

SYSTEM_PROMPT = """You are an expert in evaluating the quality of factual claims.
Your task is to assess whether a given claim is a high-quality, assessable claim
based on a strict set of criteria. You must be precise and consistent."""

GUIDEBOOK = """# Claim Quality Assessment: Is the claim assessable?

## Definition
A high-quality claim is a clear, self-contained, verifiable statement that accurately reflects information in a source article.

## Key Qualities of a High-Quality Claim

### Self-Contained and Clear
- Includes all essential context (who, what, where, when) needed to understand the assertion
- Does not rely on the reader having read the source article
- Uses specific entities rather than ambiguous references or unexplained pronouns

### Verifiable
- States an objective fact that could be proven true or false
- Avoids opinions, subjective statements, predictions, or hypotheticals
- Is specific enough to be verified with evidence

## Assessment Scale
- **Yes (1)** = High Quality: The claim meets ALL criteria above.
- **No (0)** = Not High Quality: The claim fails to meet one or more criteria.

## Examples

### Yes (High Quality):
- "According to a NASA report released on May 12, 2023, the James Webb Space Telescope discovered water vapor in the atmosphere of exoplanet K2-18b."
- "Apple announced its fourth-quarter revenue of $89.5 billion on October 28, which represents a 2.1% increase from the same period last year."
- "The World Health Organization declared COVID-19 a global pandemic on March 11, 2020."

### No (Not High Quality):
- "NASA scientists recently found something on an exoplanet." (Too vague, lacks specificity)
- "He said the company would face consequences." (Unclear entities, missing context)
- "Experts believe this might be the best approach to solving the crisis." (Subjective, prediction)
- "The report shows concerning trends in several key metrics." (Vague, lacks specific facts)
- "According to sources, the deal failed because they couldn't agree on terms." (Ambiguous references)

## Edge Cases
- Year missing: NO
- Mixed language: NO
- Only quote in a different language: YES
- No date: depends — for super specific events (bridge collapsing) YES; for non-specific events NO
- Imprecise punctuation: YES
- Incomplete sentence: NO"""

EVAL_TEMPLATE = """---

## Claim to evaluate:
{claim}

## Instructions:
Analyze the claim against ALL criteria above. Provide your reasoning, then give your binary assessment.

<REASONING>
Step-by-step analysis of whether this claim is self-contained, clear, and verifiable.
</REASONING>

<ASSESSMENT>
is_checkable: [1/0]
</ASSESSMENT>"""


def build_fewshot_block(examples_df, include_notes=False):
    """Build a few-shot examples block from consensus examples."""
    lines = ["\n## Annotated Examples from Expert Labelers\n"]
    lines.append("Below are real claims evaluated by expert annotators. Use these as reference:\n")
    for _, row in examples_df.iterrows():
        label = row['is_checkable_ruggero']
        label_str = "1 (High Quality)" if label == 1 else "0 (Not High Quality)"
        lines.append(f"**Claim:** {row['claim']}")
        lines.append(f"**is_checkable:** {label_str}")
        if include_notes:
            notes_r = str(row.get('notes_ruggero', ''))
            notes_i = str(row.get('notes_ilaria', ''))
            if notes_r and notes_r != 'nan':
                lines.append(f"**Note:** {notes_r}")
            elif notes_i and notes_i != 'nan':
                lines.append(f"**Note:** {notes_i}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(claim, fewshot_block=""):
    """Build the full evaluation prompt."""
    parts = [GUIDEBOOK]
    if fewshot_block:
        parts.append(fewshot_block)
    parts.append(EVAL_TEMPLATE.format(claim=claim))
    return "\n".join(parts)


def strip_thinking_tags(text):
    """Strip <think>...</think> tags from thinking models."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_response(answer):
    """Parse LLM response for is_checkable binary evaluation."""
    answer = strip_thinking_tags(answer)
    reasoning = ""
    is_checkable = None

    reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', answer, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    assessment_match = re.search(r'<ASSESSMENT>(.*?)</ASSESSMENT>', answer, re.DOTALL)
    if assessment_match:
        checkable_match = re.search(r'is_checkable:\s*(\d)', assessment_match.group(1))
        if checkable_match:
            is_checkable = int(checkable_match.group(1))

    if is_checkable is None:
        if re.search(r'\bis_checkable\s*[:=]\s*1\b', answer, re.IGNORECASE):
            is_checkable = 1
        elif re.search(r'\bis_checkable\s*[:=]\s*0\b', answer, re.IGNORECASE):
            is_checkable = 0
        else:
            raise ValueError("Could not parse is_checkable value from response")

    return reasoning, is_checkable


def evaluate_single(client, model, claim, fewshot_block, extra_body, retries=3, max_tokens=2048):
    """Evaluate a single claim with retries."""
    prompt = build_prompt(claim, fewshot_block)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            answer = resp.choices[0].message.content
            reasoning, is_checkable = parse_response(answer)
            return is_checkable, reasoning
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                logger.error(f"Failed after {retries} attempts: {e}")
                return -1, f"Failed: {e}"


def evaluate_config(client, model, test_df, fewshot_block, extra_body, config_name, max_workers=20, max_tokens=2048):
    """Evaluate all test claims with a given few-shot config."""
    results = []
    with tqdm(total=len(test_df), desc=config_name, leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_single, client, model,
                    row['claim'], fewshot_block, extra_body,
                    max_tokens=max_tokens
                ): idx for idx, (_, row) in enumerate(test_df.iterrows())
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                is_checkable, reasoning = future.result()
                results.append({
                    'idx': idx,
                    'is_checkable_llm': is_checkable,
                    'reasoning': reasoning,
                })
                pbar.update(1)
                success = sum(1 for r in results if r['is_checkable_llm'] >= 0)
                pbar.set_postfix(success=f"{100*success/len(results):.0f}%")

    results_df = pd.DataFrame(results).sort_values('idx').reset_index(drop=True)
    return results_df


def compute_metrics(test_df, results_df):
    """Compute agreement metrics."""
    llm = results_df['is_checkable_llm'].values
    valid_llm = llm >= 0

    metrics = {
        'n_valid': int(valid_llm.sum()),
        'n_failed': int((~valid_llm).sum()),
        'llm_checkable': int((llm[valid_llm] == 1).sum()),
        'llm_not_checkable': int((llm[valid_llm] == 0).sum()),
    }
    for name, col in [('ruggero', 'is_checkable_ruggero'), ('ilaria', 'is_checkable_ilaria')]:
        human = test_df[col].values
        # Only compare where both LLM and human have valid values
        valid = valid_llm & ~pd.isna(human)
        h = human[valid].astype(int)
        l = llm[valid].astype(int)
        if len(h) == 0:
            metrics[f'kappa_vs_{name}'] = 0.0
            metrics[f'acc_vs_{name}'] = 0.0
            metrics[f'fp_vs_{name}'] = 0
            metrics[f'fn_vs_{name}'] = 0
            continue
        kappa = cohen_kappa_score(h, l)
        acc = accuracy_score(h, l)
        cm = confusion_matrix(h, l, labels=[0, 1])
        metrics[f'kappa_vs_{name}'] = round(kappa, 4)
        metrics[f'acc_vs_{name}'] = round(acc, 4)
        metrics[f'fp_vs_{name}'] = int(cm[0, 1])
        metrics[f'fn_vs_{name}'] = int(cm[1, 0])
    return metrics


def select_examples(cal_consensus, n_total, strategy='balanced', seed=42):
    """Select few-shot examples using different strategies."""
    rng = np.random.RandomState(seed)
    checkable = cal_consensus[cal_consensus['is_checkable_ruggero'] == 1]
    not_checkable = cal_consensus[cal_consensus['is_checkable_ruggero'] == 0]

    if strategy == 'balanced':
        n_each = n_total // 2
        n_c = min(n_each, len(checkable))
        n_nc = min(n_each, len(not_checkable))
        sel_c = checkable.sample(n_c, random_state=rng)
        sel_nc = not_checkable.sample(n_nc, random_state=rng)
        # If we have leftover budget (one class smaller), fill from the other
        used = n_c + n_nc
        if used < n_total and used < len(cal_consensus):
            remaining = cal_consensus.drop(sel_c.index).drop(sel_nc.index)
            extra = remaining.sample(min(n_total - used, len(remaining)), random_state=rng)
            return pd.concat([sel_nc, sel_c, extra]).sample(frac=1, random_state=rng)
        return pd.concat([sel_nc, sel_c]).sample(frac=1, random_state=rng)

    elif strategy == 'fp_heavy':
        # More not-checkable examples to reduce false positives
        n_nc = min(int(n_total * 0.7), len(not_checkable))
        n_c = n_total - n_nc
        sel_nc = not_checkable.sample(n_nc, random_state=rng)
        sel_c = checkable.sample(min(n_c, len(checkable)), random_state=rng)
        return pd.concat([sel_nc, sel_c]).sample(frac=1, random_state=rng)

    elif strategy == 'all_not_checkable':
        return not_checkable.sample(min(n_total, len(not_checkable)), random_state=rng)

    elif strategy == 'diverse':
        # Pick examples that are most borderline/interesting (shortest claims)
        cal_sorted = cal_consensus.copy()
        cal_sorted['claim_len'] = cal_sorted['claim'].str.len()
        # Mix short and long claims
        short = cal_sorted.nsmallest(n_total, 'claim_len')
        n_each = n_total // 2
        sel_c = short[short['is_checkable_ruggero'] == 1].head(n_each)
        sel_nc = short[short['is_checkable_ruggero'] == 0].head(n_each)
        result = pd.concat([sel_nc, sel_c])
        if len(result) < n_total:
            remaining = cal_consensus.drop(result.index).sample(
                n_total - len(result), random_state=rng
            )
            result = pd.concat([result, remaining])
        return result.sample(frac=1, random_state=rng)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


STRICT_SYSTEM_PROMPT = """You are an expert in evaluating the quality of factual claims.
Your task is to assess whether a given claim is a high-quality, assessable claim
based on a strict set of criteria. You must be precise and consistent.

IMPORTANT: Be STRICT in your evaluation. When in doubt, mark the claim as NOT checkable (0).
A claim must be FULLY self-contained, COMPLETELY clear, and UNAMBIGUOUSLY verifiable to score 1.
Even minor issues with clarity, specificity, or verifiability should result in a 0."""


def main():
    parser = argparse.ArgumentParser(description='Few-shot is_checkable experiment')
    parser.add_argument('--annotated_csv', default='data/samples/labeling_sample_200_annotated.csv')
    parser.add_argument('--model', default='Qwen/Qwen3-32B')
    parser.add_argument('--port', type=int, default=7501)
    parser.add_argument('--max_workers', type=int, default=20)
    parser.add_argument('--output_dir', default='results/fewshot_experiment')
    parser.add_argument('--round', type=int, default=1, help='Experiment round (1=baseline, 2=refined)')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for generation')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    df = pd.read_csv(args.annotated_csv)
    cal = df[df['split'] == 'calibration'].copy()
    test = df[df['split'] == 'test'].copy().reset_index(drop=True)

    # Consensus calibration examples
    cal_consensus = cal[cal['is_checkable_ruggero'] == cal['is_checkable_ilaria']].copy()
    print(f"Calibration consensus: {len(cal_consensus)} examples "
          f"({(cal_consensus.is_checkable_ruggero==1).sum()} checkable, "
          f"{(cal_consensus.is_checkable_ruggero==0).sum()} not checkable)")
    print(f"Test set: {len(test)} claims")

    # Setup client
    client = openai.OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="local")
    extra_body = {}
    model_lower = args.model.lower()
    # Disable thinking for Qwen3 models (not QwQ which is Qwen2.5-based)
    if "qwen3" in model_lower and "qwq" not in model_lower:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    # Auto-increase max_tokens for thinking models
    is_thinking = any(t in model_lower for t in ["qwq", "deepseek-r1", "magistral"])
    if is_thinking and args.max_tokens <= 2048:
        args.max_tokens = 4096
        print(f"Auto-increased max_tokens to {args.max_tokens} for thinking model")

    # Define experiment configurations
    # Round 1: baseline exploration
    # Round 2: refined based on round 1 findings (diverse works, strict helps, more shots hurt)
    if args.round == 1:
        configs = [
            # (name, n_shots, strategy, include_notes, use_strict)
            ("zero_shot", 0, None, False, False),
            ("4shot_balanced", 4, "balanced", False, False),
            ("8shot_balanced", 8, "balanced", False, False),
            ("16shot_balanced", 16, "balanced", False, False),
            ("8shot_fp_heavy", 8, "fp_heavy", False, False),
            ("16shot_fp_heavy", 16, "fp_heavy", False, False),
            ("8shot_all_nc", 8, "all_not_checkable", False, False),
            ("16shot_balanced_notes", 16, "balanced", True, False),
            ("8shot_diverse", 8, "diverse", False, False),
        ]
    elif args.round == 2:
        configs = [
            # Build on round 1 winner (8shot_diverse) + try strict prompt + different counts
            ("zero_shot_strict", 0, None, False, True),
            ("4shot_diverse", 4, "diverse", False, False),
            ("6shot_diverse", 6, "diverse", False, False),
            ("8shot_diverse", 8, "diverse", False, False),
            ("10shot_diverse", 10, "diverse", False, False),
            ("12shot_diverse", 12, "diverse", False, False),
            ("4shot_diverse_strict", 4, "diverse", False, True),
            ("6shot_diverse_strict", 6, "diverse", False, True),
            ("8shot_diverse_strict", 8, "diverse", False, True),
            ("10shot_diverse_strict", 10, "diverse", False, True),
            ("8shot_diverse_notes", 8, "diverse", True, False),
            ("8shot_diverse_strict_notes", 8, "diverse", True, True),
        ]
    else:
        # Round 3: scale up consensus few-shots with strict prompt (R2 winner pattern)
        configs = [
            # Re-run R2 winner for reproducibility
            ("10shot_diverse_strict", 10, "diverse", False, True),
            # Scale up diverse+strict
            ("14shot_diverse_strict", 14, "diverse", False, True),
            ("18shot_diverse_strict", 18, "diverse", False, True),
            ("24shot_diverse_strict", 24, "diverse", False, True),
            # Try balanced+strict at higher counts
            ("16shot_balanced_strict", 16, "balanced", False, True),
            ("24shot_balanced_strict", 24, "balanced", False, True),
            ("32shot_balanced_strict", 32, "balanced", False, True),
            # Try ALL consensus examples (87 total, unbalanced)
            ("all87_consensus_strict", 87, "balanced", False, True),
            # FP-heavy at higher counts with strict
            ("16shot_fp_heavy_strict", 16, "fp_heavy", False, True),
            ("24shot_fp_heavy_strict", 24, "fp_heavy", False, True),
            # All not-checkable + strict (max=27)
            ("27shot_all_nc_strict", 27, "all_not_checkable", False, True),
            # Diverse with notes at higher counts
            ("16shot_diverse_strict_notes", 16, "diverse", True, True),
            ("24shot_diverse_strict_notes", 24, "diverse", True, True),
        ]

    all_results = []

    for config_name, n_shots, strategy, include_notes, use_strict in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({n_shots} shots, strategy={strategy}, strict={use_strict})")
        print(f"{'='*60}")

        if n_shots > 0:
            examples = select_examples(cal_consensus, n_shots, strategy)
            fewshot_block = build_fewshot_block(examples, include_notes=include_notes)
        else:
            fewshot_block = ""

        # Override system prompt if strict mode
        if use_strict:
            orig_sys = SYSTEM_PROMPT
            # Temporarily monkey-patch for this config
            import types
            def _make_eval(sys_prompt, max_tok):
                def _eval(client, model, claim, fewshot_block, extra_body, retries=3):
                    prompt = build_prompt(claim, fewshot_block)
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    for attempt in range(retries):
                        try:
                            resp = client.chat.completions.create(
                                model=model, messages=messages,
                                temperature=0.1, max_tokens=max_tok,
                                extra_body=extra_body,
                            )
                            answer = resp.choices[0].message.content
                            reasoning, is_checkable = parse_response(answer)
                            return is_checkable, reasoning
                        except Exception as e:
                            if attempt < retries - 1:
                                time.sleep(1)
                            else:
                                return -1, f"Failed: {e}"
                return _eval

            eval_fn = _make_eval(STRICT_SYSTEM_PROMPT, args.max_tokens)
            # Run with custom eval function
            start = time.time()
            results = []
            with tqdm(total=len(test), desc=config_name, leave=True) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = {
                        executor.submit(
                            eval_fn, client, args.model,
                            row['claim'], fewshot_block, extra_body
                        ): idx for idx, (_, row) in enumerate(test.iterrows())
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        is_checkable, reasoning = future.result()
                        results.append({'idx': idx, 'is_checkable_llm': is_checkable, 'reasoning': reasoning})
                        pbar.update(1)
            results_df = pd.DataFrame(results).sort_values('idx').reset_index(drop=True)
        else:
            start = time.time()
            results_df = evaluate_config(
                client, args.model, test, fewshot_block, extra_body,
                config_name, max_workers=args.max_workers, max_tokens=args.max_tokens
            )
        elapsed = time.time() - start

        metrics = compute_metrics(test, results_df)
        metrics['config'] = config_name
        metrics['n_shots'] = n_shots
        metrics['strategy'] = strategy or 'none'
        metrics['include_notes'] = include_notes
        metrics['use_strict'] = use_strict
        metrics['elapsed_s'] = round(elapsed, 1)

        print(f"\n  Results ({elapsed:.1f}s):")
        print(f"  Kappa vs Ruggero: {metrics['kappa_vs_ruggero']:.3f} | Acc: {metrics['acc_vs_ruggero']:.3f} | FP: {metrics['fp_vs_ruggero']} | FN: {metrics['fn_vs_ruggero']}")
        print(f"  Kappa vs Ilaria:  {metrics['kappa_vs_ilaria']:.3f} | Acc: {metrics['acc_vs_ilaria']:.3f} | FP: {metrics['fp_vs_ilaria']} | FN: {metrics['fn_vs_ilaria']}")

        all_results.append(metrics)

        # Save per-config results
        results_df.to_csv(f"{args.output_dir}/{config_name}_{timestamp}.csv", index=False)

    # Save comparison table
    comparison = pd.DataFrame(all_results)
    comparison_file = f"{args.output_dir}/comparison_{timestamp}.csv"
    comparison.to_csv(comparison_file, index=False)

    # Print final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    print(comparison[['config', 'n_shots', 'strategy', 'kappa_vs_ruggero', 'kappa_vs_ilaria',
                       'acc_vs_ruggero', 'acc_vs_ilaria', 'fp_vs_ruggero', 'fn_vs_ruggero']].to_string(index=False))
    print(f"\nResults saved to {comparison_file}")

    # Return best config
    comparison['avg_kappa'] = (comparison['kappa_vs_ruggero'] + comparison['kappa_vs_ilaria']) / 2
    best = comparison.loc[comparison['avg_kappa'].idxmax()]
    print(f"\nBest config: {best['config']} (avg kappa: {best['avg_kappa']:.3f})")

    return comparison


if __name__ == '__main__':
    main()
