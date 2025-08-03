import re
import nltk
from nltk.sem import logic
from nltk.sem import Expression
import warnings
import functools
import os

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
    prover = nltk.Prover9(30)  # 10-second timeout
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
    print(f"DEBUG[utils.py]: evaluate called with premises={premises_tuple}, conclusion={conclusion}")
    try:
        # Convert tuple back to list (since tuples are hashable for caching)
        premises = list(premises_tuple)

        # Extract all predicate names from the premises and conclusion
        def extract_predicates(formula):
            # Find all tokens that look like predicates (uppercase followed by parenthesis)
            predicate_pattern = r'([A-Z][a-zA-Z0-9_]*)\('
            return set(re.findall(predicate_pattern, formula))
        
        # Fix arity conflicts by renaming constants with the same name as predicates
        def fix_arity_conflicts(formula, predicates):
            # For each predicate, find instances where it's used as a constant
            for pred in predicates:
                # Replace when used as a constant (not followed by opening parenthesis)
                # but only when it's a whole word (bounded by word boundaries)
                formula = re.sub(r'\b' + pred + r'\b(?!\()', pred + '_CONST', formula)
            return formula
        
        # Create debug directory
        os.makedirs("debug", exist_ok=True)
        
        # First, extract all predicate names from premises and conclusion
        all_predicates = set()
        for p in premises:
            all_predicates.update(extract_predicates(p))
        all_predicates.update(extract_predicates(conclusion))
        
        # Debug: log the detected predicates
        with open("debug/debug_predicates.txt", "w") as f:
            f.write("Detected predicates:\n")
            for p in sorted(all_predicates):
                f.write(f"  {p}\n")
        
        # Now fix arity conflicts in all formulas
        fixed_premises = [fix_arity_conflicts(p, all_predicates) for p in premises]
        fixed_conclusion = fix_arity_conflicts(conclusion, all_predicates)
        
        # Debug: log the fixed formulas
        with open("debug/debug_fixed_formulas.txt", "w") as f:
            f.write("Original premises:\n")
            for p in premises:
                f.write(f"  {p}\n")
            f.write("\nFixed premises:\n")
            for p in fixed_premises:
                f.write(f"  {p}\n")
            f.write(f"\nOriginal conclusion: {conclusion}\n")
            f.write(f"Fixed conclusion: {fixed_conclusion}\n")
        
        # Continue with your existing code, but use the fixed versions
        premises = [reformat_fol(p) for p in fixed_premises]
        conclusion = reformat_fol(fixed_conclusion)

        # Debug: print reformatted premises and conclusion
        print(f"[DEBUG] Premises: {premises}")
        print(f"[DEBUG] Conclusion: {conclusion}")

        # Validate that premises and conclusion are non-empty
        if not conclusion or not all(premises):
            warnings.warn("Empty premise or conclusion detected")
            print("[DEBUG] Empty premise or conclusion detected")
            return "Uncertain"

        c = read_expr(conclusion)
        p_list = []
        for p in premises:
            p_list.append(read_expr(p))
        
        # Debug: print parsed expressions
        print(f"[DEBUG] Parsed conclusion: {c}")
        print(f"[DEBUG] Parsed premises: {p_list}")

        # Add timeout handling
        truth_value = prover.prove(c, p_list)
        print(f"[DEBUG] Prover result for conclusion: {truth_value}")
        if truth_value:
            return "True"
        else:
            neg_c = read_expr("-(" + conclusion + ")")
            negation_true = prover.prove(neg_c, p_list)
            print(f"[DEBUG] Prover result for negated conclusion: {negation_true}")
            if negation_true:
                return "False"
            else:
                return "Uncertain"
    except Exception as e:
        warnings.warn(f"Error in FOL evaluation: {e}")
        print(f"[DEBUG] Exception in evaluate: {e}")
        
        # Save error details to debug file
        with open("debug/debug_evaluate_error.txt", "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Premises:\n")
            for p in premises_tuple:
                f.write(f"  {p}\n")
            f.write(f"Conclusion: {conclusion}\n")
            
        return "Uncertain"  # Default to uncertain on error
