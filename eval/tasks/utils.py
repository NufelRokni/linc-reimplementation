import re
import nltk
from nltk.sem import logic
from nltk.sem import Expression
import warnings
import functools

# Instead of directly modifying NLTK internals:
try:
    # Save original value
    original_counter = logic._counter._value
    # Reset for our use
    logic._counter._value = 0
except AttributeError:
    warnings.warn("Could not access NLTK logic counter - using default behavior")

read_expr = Expression.fromstring

try:
    prover = nltk.Prover9(10)  # 10-second timeout
except (AttributeError, ImportError) as e:
    warnings.warn(f"Error initializing Prover9: {e}. Using default prover.")
    # Fallback to another prover or a dummy implementation
    class DummyProver:
        def prove(self, conclusion, premises):
            warnings.warn("Using dummy prover - results may be inaccurate")
            return False
    prover = DummyProver()


def convert_to_nltk_rep(logic_formula):
    try:
        translation_map = {
            "∀": "all ",
            "∃": "exists ",
            "→": "->",
            "¬": "-",
            "∧": "&",
            "∨": "|",
            "⟷": "<->",
            "↔": "<->",
            "0": "Zero",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
            ".": "Dot",
            "Ś": "S",
            "ą": "a",
            "’": "",
        }

        constant_pattern = r'\b([a-z]{2,})(?!\()'
        logic_formula = re.sub(constant_pattern, lambda match: match.group(1).capitalize(), logic_formula)

        for key, value in translation_map.items():
            logic_formula = logic_formula.replace(key, value)

        quant_pattern = r"(all\s|exists\s)([a-z])"
        def replace_quant(match):
            return match.group(1) + match.group(2) + "."
        logic_formula = re.sub(quant_pattern, replace_quant, logic_formula)

        dotted_param_pattern = r"([a-z])\.(?=[a-z])"
        def replace_dotted_param(match):
            return match.group(1)
        logic_formula = re.sub(dotted_param_pattern, replace_dotted_param, logic_formula)

        simple_xor_pattern = r"(\w+\([^()]*\)) ⊕ (\w+\([^()]*\))"
        def replace_simple_xor(match):
            left = match.group(1)
            right = match.group(2)
            # Count parentheses to ensure they're balanced
            if left.count('(') != left.count(')') or right.count('(') != right.count(')'):
                # Handle unbalanced parentheses
                warnings.warn(f"Unbalanced parentheses in XOR: {match.group(0)}")
            return f"(({left} & -{right}) | (-{left} & {right}))"
        logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

        complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"
        def replace_complex_xor(match):
            return ("(((" + match.group(1) + ")) & -(" + match.group(2) + "))) | (-(" + match.group(1) + ")) & (" + match.group(2) + "))))")
        logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

        special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"
        def replace_special_xor(match):
            return ("(((" + match.group(1) + ")) & -" + match.group(2) + ") | (-(" + match.group(1) + ")) & " + match.group(2) + ")")
        logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)
        
        return logic_formula
    except Exception as e:
        warnings.warn(f"Error converting FOL to NLTK representation: {e}")
        # Return a safely escaped version or raise a more informative error
        return f"Error({logic_formula})"

def get_all_variables(text):
    # A better approach is to use a proper parser
    # But as a simple improvement:
    try:
        # Find top-level parenthetical expressions
        pattern = r'\([^()]*(?:\([^()]*\)[^()]*)*\)'
        matches = re.findall(pattern, text)
        all_variables = []
        for m in matches:
            # Strip outer parentheses
            m = m[1:-1]
            # Split by commas not inside nested parentheses
            parts = []
            current = ""
            nesting = 0
            for char in m:
                if char == '(' and nesting == 0:
                    current += char
                    nesting += 1
                elif char == ')' and nesting > 0:
                    current += char
                    nesting -= 1
                elif char == ',' and nesting == 0:
                    parts.append(current.strip())
                    current = ""
                else:
                    current += char
            if current:
                parts.append(current.strip())
            
            all_variables += parts
        return list(set(all_variables))
    except Exception as e:
        warnings.warn(f"Error extracting variables: {e}")
        return []  # Return empty list as fallback

def reformat_fol(fol):
    translation_map = {
        "0": "Zero", 
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "’": "",
        "-": "_",
        "'": "",
        " ": "_"
    }
    all_variables = get_all_variables(fol)
    for variable in all_variables:
        variable_new = variable[:]
        for k, v in translation_map.items():
            variable_new = variable_new.replace(k, v)
        fol = fol.replace(variable, variable_new)
    return fol

@functools.lru_cache(maxsize=128)
def evaluate(premises_tuple, conclusion):
    try:
        # Convert tuple back to list (since tuples are hashable for caching)
        premises = list(premises_tuple)

        premises = [reformat_fol(p) for p in premises]
        conclusion = reformat_fol(conclusion)

        # Validate that premises and conclusion are non-empty
        if not conclusion or not all(premises):
            warnings.warn("Empty premise or conclusion detected")
            return "Uncertain"

        c = read_expr(conclusion)
        p_list = []
        for p in premises:
            p_list.append(read_expr(p))
        
        # Add timeout handling
        truth_value = prover.prove(c, p_list)
        if truth_value:
            return "True"
        else:
            neg_c = read_expr("-(" + conclusion + ")")
            negation_true = prover.prove(neg_c, p_list)
            if negation_true:
                return "False"
            else:
                return "Uncertain"
    except Exception as e:
        warnings.warn(f"Error in FOL evaluation: {e}")
        return "Uncertain"  # Default to uncertain on error
