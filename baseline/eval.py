import json
import os
from tqdm import tqdm

from .utils import load_data_set, build_prompt, parse_label

def eval_baseline(model, args):
    """
    Run baseline evaluation.
    Args:
        model: The language model object.
        args:  The parsed command-line arguments.
    """
    print("Running baseline evaluation...")

    # 1. Load the dataset
    data = load_data_set(args.task)

    # 2. Run inference and gather results
    results = []
    correct_predictions = 0
    total_predictions = 0

    for ex in tqdm(data, desc="Evaluating"):
        try:
            # 2a. Build the prompt for the current example
            prompt = build_prompt(ex, args)

            # 2b. Generate a response from the model
            # Assuming model.generate takes a prompt and returns a string.
            # Additional generation parameters can be passed from args if needed.
            output = model.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            # 2c. Parse the model's output to get the final label
            predicted_label = parse_label(output)
            ground_truth_label = ex.get("answer") # Assuming the ground truth is in the 'answer' field

            # 2d. Store results and check for correctness
            is_correct = (predicted_label == ground_truth_label)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            results.append({
                "id": ex.get("id"),
                "prompt": prompt,
                "model_output": output,
                "predicted_label": predicted_label,
                "ground_truth_label": ground_truth_label,
                "is_correct": is_correct
            })
        except Exception as e:
            print(f"\nAn error occurred while processing an example: {e}")
            continue

    # 3. Calculate and print metrics
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nEvaluation Finished.")
        print(f"Total Examples: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo examples were processed.")
        return

    # 4. Save the detailed results to a file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "evaluation_results.json")
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Detailed results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
