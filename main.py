from __future__ import annotations

from arguments import build_arg_parser
from baseline.eval import eval_baseline
from scratchpad.eval import eval_scratchpad
from cot.eval import eval_cot
from linc.eval import eval_linc


def main() -> None:
    args = build_arg_parser().parse_args()
    
    model = LMModel(args.model, precision=args.precision, device=args.device)
    
    if args.mode == "baseline":
        eval_baseline(model, args)
    elif args.mode == "scratchpad":
        eval_scratchpad(model, args)
    elif args.mode == "cot":
        eval_cot(model, args)
    elif args.mode == "linc":
        eval_linc(model, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":  # pragma: no cover
    main()
