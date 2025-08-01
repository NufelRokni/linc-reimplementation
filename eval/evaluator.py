import json
import os

from abc import abstractmethod, ABC
import warnings
from . import tasks
from .generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The task you are about to use executes untrusted model-generated code.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator(ABC):
    def __init__(self, args):
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    @abstractmethod
    def generate_text(self, task_name):
        pass

    def evaluate(self, task_name):
        print(f"Evaluating task: {task_name}")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            print(f"eval.evaluate: {task_name} requires code execution, but allow_code_execution is False")
            raise ValueError(_WARNING)

        generations_prc, generations_raw, references = self.generate_text(task_name)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [samples[: self.args.n_samples] for samples in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )

        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    try:
                        os.makedirs(os.path.dirname(self.args.save_generations_raw_path), exist_ok=True)
                        with open(self.args.save_generations_raw_path, "w") as fp:
                            json.dump(generations_raw, fp)
                            print("raw generations were saved")
                    except Exception as e:
                        warnings.warn(f"Failed to save raw generations: {e}")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                        print("processed generations were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")

            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            return task.process_results(generations_prc, references)


class HFEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        super().__init__(args)
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, task_name):
        print(f"Generating text for task: {task_name}")
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        if len(dataset) == 0:
            warnings.warn(f"Empty dataset for task {task_name}")
            return [], [], []
            
        n_tasks = min(len(dataset), self.args.limit) if self.args.limit else len(dataset)
        try:
            generations_prc, generations_raw = parallel_generations(
                task,
                dataset,
                self.accelerator,
                self.model,
                self.tokenizer,
                n_tasks=n_tasks,
                args=self.args,
            )
            references = []
            for i in range(n_tasks):
                try:
                    references.append(task.get_reference(dataset[i]))
                except Exception as e:
                    warnings.warn(f"Failed to get reference for sample {i}: {e}")
                    references.append(None)  # Or an appropriate default
            return generations_prc, generations_raw, references
        except Exception as e:
            warnings.warn(f"Error during generation for task {task_name}: {e}")
            return [], [], []
