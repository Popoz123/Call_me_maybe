import sys
from pathlib import Path
from src.decoder import Decoder
from src.parser import InputParser, FuncDef
from typing import Any
import json
import argparse


if __name__ == "__main__":
    f_parser = argparse.ArgumentParser()

    h_str: str = "path to function definitions"
    f_parser.add_argument(
        "-fd", "--functions_definition",
        default="data/input/functions_definition.json",
        help=h_str
    )

    h_str = "path to prompts"
    f_parser.add_argument(
        "-i", "--input",
        default="data/input/function_calling_tests.json",
        help=h_str
    )

    h_str = "path to output location"
    f_parser.add_argument(
        "-o", "--output",
        default="data/output/function_calling_results.json",
        help=h_str
    )

    h_str = "Enable animated visualization"
    f_parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help=h_str
    )

    args = f_parser.parse_args()

    parser: InputParser = InputParser()
    dcl: Decoder = Decoder(visualize=args.visualize)

    try:
        parser.parse_defs(args.functions_definition)
        parser.parse_prompts(args.input)
    except (OSError, ValueError) as e:
        print(e)
        sys.exit(1)

    f_inputs: list[str] = parser.format_input()
    results: list[dict[str, Any]] = []

    if dcl.llm is None:
        print("LLM instance not found")
        sys.exit(1)

    for f_prompt in f_inputs:
        fn_name: tuple[FuncDef, list[int]] = dcl.get_fn_name(
            parser.functions, f_prompt
        )
        res_ids: list[int] = dcl.get_params(fn_name[0], fn_name[1])

        res_str: str = dcl.llm.decode(res_ids)
        _, res = res_str.split("JSON output:", 1)
        try:
            item: dict[str, Any] = json.loads(res.strip())
            results.append(item)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}", file=sys.stderr)

    f_path: Path = Path(args.output)
    f_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        f.write("\n")
