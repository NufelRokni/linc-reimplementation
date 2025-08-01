import json
import math
import warnings

from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate.utils import set_seed

from .utils import TokenizedDataset, complete_code


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = 0 if start_length is None else start_length  # Safe default
        self.eof_strings = eof_strings or ["</s>"]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.generations_path:
        try:
            with open(args.generations_path) as fp:
                generations = json.load(fp)
                if not isinstance(generations, list) or (len(generations) > 0 and not isinstance(generations[0], list)):
                    raise ValueError(f"Invalid generations format: expected list of lists, got {type(generations)}")
                if accelerator.is_main_process:
                    print(f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates")
            return generations[:n_tasks]
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            if accelerator.is_main_process:
                print(f"Error loading generations: {e}")
            raise

    set_seed(args.seed, device_specific=True)

    # Use *max_new_tokens* instead of *max_length* so the model can stop early
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    if task.stop_words:
        # This should be initialized with the prompt length to only check generated content
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )
        # Later in complete_code, we set the actual start_length based on input context

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")

    # Calculate n_copies safely (avoid zero and cover all samples)
    n_copies = args.n_samples // args.batch_size
    if n_copies == 0 or (args.n_samples % args.batch_size != 0):
        n_copies += 1

    # Ensure even distribution across processes (no device gets empty batch)
    total_prompts = n_tasks * n_copies
    num_procs = accelerator.num_processes
    if total_prompts % num_procs != 0:
        import math

        # Find minimal copies to balance across all processes
        lcm_val = lcm(n_tasks, num_procs)
        min_copies = lcm_val // n_tasks
        if n_copies < min_copies:
            n_copies = min_copies
        elif n_copies % min_copies != 0:
            n_copies = ((n_copies // min_copies) + 1) * min_copies
        if accelerator.is_main_process:
            warnings.warn(
                f"Adjusted n_copies to {n_copies} for even distribution across {num_procs} devices."
            )

    if accelerator.is_main_process:
        print(f"Using n_copies={n_copies} (each task will be run {n_copies} time(s) to generate up to {n_copies * args.batch_size} outputs, trimming to {args.n_samples}).")

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    model, ds_loader = accelerator.prepare(model, ds_loader)
    try:
        generations_prc, generations_raw = complete_code(
            task,
            accelerator,
            model,
            tokenizer,
            ds_loader,
            n_tasks=n_tasks,
            batch_size=args.batch_size,
            prefix=args.prefix,
            postprocess=args.postprocess,
            **gen_kwargs,
        )
        return generations_prc, generations_raw
    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error during code generation: {e}")
        raise
