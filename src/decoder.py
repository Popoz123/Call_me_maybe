import json
try:
    import numpy as np
    import numpy.typing as npt
    from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
except ModuleNotFoundError as e:
    print(f"Missing modules - use make install: {e}")
    exit()
from src.parser import FuncDef
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
import sys
import time


class Decoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    visualize: bool = False
    model_name: str = "Qwen/Qwen3-0.6B"
    llm: Small_LLM_Model | None = None
    number_t: set[int] = Field(default_factory=set)
    str_mask: npt.NDArray[np.float32] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.llm is None:
            self.llm = Small_LLM_Model(model_name=self.model_name)
        self._load_vocab()

    @staticmethod
    def _is_num_t(t_str: str) -> bool:
        s = t_str.strip()
        if not s:
            return False
        return all(c in "0123456789.-" for c in s)

    def _load_vocab(self) -> None:
        assert self.llm is not None
        vocab_p: str = self.llm.get_path_to_vocab_file()
        try:
            with open(vocab_p, 'r') as f:
                token_to_id: dict[str, int] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise Exception(f"Vocabulary file corrupted: {e}")

        in_chars: set[str] = {'\n', '\r', '"', 'Ċ', 'č'}
        quote_d: int = self.llm.encode('"')[0][0].item()
        quote_comma: int = self.llm.encode('",')[0][0].item()
        quote_brace: int = self.llm.encode('"}')[0][0].item()

        vocab_s: int = len(self.llm.get_logits_from_input_ids([quote_d]))
        self.str_mask = np.full(vocab_s, -np.inf, dtype=np.float32)
        self.number_t = set()

        for t_str, t_id in token_to_id.items():
            if self._is_num_t(t_str):
                self.number_t.add(t_id)
            if not any(c in in_chars for c in t_str):
                self.str_mask[t_id] = 0.0

        if not self.number_t:
            raise RuntimeError("number_t is empty — vocab loading failed")

        self.str_mask[quote_d] = 0.0
        self.str_mask[quote_comma] = 0.0
        self.str_mask[quote_brace] = 0.0

    @staticmethod
    def cl_lines(n: int) -> None:
        for _ in range(n):
            sys.stdout.write("\033[F\033[K")
        sys.stdout.flush()

    def get_fn_name(self, funcs: list[FuncDef],
                    prompt: str) -> tuple[FuncDef, list[int]]:

        assert self.llm is not None
        assert self.str_mask is not None
        delim: int = self.llm.encode('"')[0][0]
        candidates: list[tuple[FuncDef, list[int]]] = [
            (func, self.llm.encode(func.name)[0].tolist() + [delim])
            for func in funcs
        ]
        c_input_ids: list[int] = self.llm.encode(prompt)[0].tolist()
        i: int = 0
        logits: list[float] = []

        ln_printed: int = 0

        while len(candidates) > 1:
            valid_t: set[int] = {token[i] for _, token in candidates}
            logits = self.llm.get_logits_from_input_ids(c_input_ids)
            best_token = max(valid_t, key=lambda t: logits[t])

            if self.visualize:
                if ln_printed > 0:
                    self.cl_lines(ln_printed)
                ln_printed = 0
                print(f"Step {i} - Evaluating {len(candidates)} candidates:")
                ln_printed += 1

                for f_name, tks in candidates:
                    next_t = tks[i]
                    sc = logits[next_t]
                    mk = "<--" if next_t == best_token else ""
                    print(f"   - {f_name.name:<35}: logit = {sc:6.2f}{mk}")
                    ln_printed += 1

                time.sleep(0.9)

            c_input_ids.append(best_token)
            candidates = [
                (f, tks) for f, tks in candidates if tks[i] == best_token
            ]
            i += 1

        if self.visualize and ln_printed > 0:
            self.cl_lines(ln_printed)

        if self.visualize:
            print(f"-> Selected function: {candidates[0][0].name}")

        _, w_tokens = candidates[0]
        c_input_ids += w_tokens[i:]
        pre_str: str = ',"parameters": {'
        c_input_ids += self.llm.encode(pre_str)[0].tolist()
        return (candidates[0][0], c_input_ids)

    def _stream(self, data: str | int | list[int], dl: float = 0.0) -> None:
        if not self.visualize:
            return

        assert self.llm is not None
        if isinstance(data, int):
            text = self.llm.decode([data])
        elif isinstance(data, list):
            text = self.llm.decode(data)
        else:
            text = data

        sys.stdout.write(text)
        sys.stdout.flush()
        if dl > 0:
            time.sleep(dl)

    def get_params(self, func: FuncDef, input_ids: list[int]) -> list[int]:
        assert self.llm is not None
        assert self.str_mask is not None
        logits_arr = np.empty(len(self.str_mask), dtype=np.float32)
        true_t: list[int] = self.llm.encode("true")[0].tolist()
        false_t: list[int] = self.llm.encode("false")[0].tolist()
        comma_d: int = self.llm.encode(",")[0][0].item()
        brace_d: int = self.llm.encode("}")[0][0].item()
        quote_d: int = self.llm.encode('"')[0][0].item()
        quote_comma: int = self.llm.encode('",')[0][0].item()
        quote_brace: int = self.llm.encode('"}')[0][0].item()
        terminators = {quote_d, quote_comma, quote_brace}

        num_params: int = len(func.parameters)

        self._stream("  Arguments: {")

        for i, (p_name, p_type) in enumerate(func.parameters.items()):
            is_last: bool = (i == num_params - 1)

            prefix = f'"{p_name}": ' if i == 0 else f', "{p_name}": '
            input_ids += self.llm.encode(prefix)[0].tolist()
            self._stream(prefix)

            logits: list[float] = []
            best_token: int

# --- BOOLEAN ---
            if p_type.type == "boolean":
                logits = self.llm.get_logits_from_input_ids(input_ids)
                if logits[true_t[0]] > logits[false_t[0]]:
                    input_ids += true_t
                    self._stream(true_t)
                else:
                    input_ids += false_t
                    self._stream(false_t)

# --- NUMBERS ---
            elif p_type.type in ("number", "integer"):
                ex_delim: int = brace_d if is_last else comma_d
                max_num_ts: int = 15
                num_step: int = 0

                logits = self.llm.get_logits_from_input_ids(input_ids)
                best_token = max(
                    self.number_t, key=lambda t: logits[t]
                )
                input_ids.append(best_token)
                self._stream(best_token, dl=0.1)

                while num_step < max_num_ts:
                    num_step += 1
                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    best_token = max(
                        self.number_t | {ex_delim}, key=lambda t: logits[t]
                    )
                    if best_token == ex_delim:
                        break
                    input_ids.append(best_token)
                    self._stream(best_token, dl=0.1)

# --- STRINGS ---
            else:
                input_ids.append(quote_d)
                self._stream(quote_d)

                seen_t: set[int] = set()
                max_ts: int = 40
                step: int = 0

                while step < max_ts:
                    step += 1

                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    logits_arr[:] = logits

                    if seen_t:
                        logits_arr[list(seen_t)] -= 10

                    masked_logits = logits_arr + self.str_mask
                    best_token = int(np.argmax(masked_logits))

                    if best_token in terminators:
                        break

                    seen_t.add(best_token)
                    input_ids.append(best_token)
                    self._stream(best_token, dl=0.02)

                if input_ids[-1] not in terminators:
                    input_ids.append(quote_d)
                    self._stream(quote_d)

        input_ids.append(brace_d)
        input_ids.append(brace_d)
        self._stream("}\n\n")
        return input_ids
