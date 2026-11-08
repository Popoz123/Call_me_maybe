import json
import sys
from parser import FuncDef, ParamType, InputParser
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    from llm_sdk.llm_sdk import Small_LLM_Model


class Decoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model_name: str = "Qwen/Qwen3-0.6B"
    llm: Small_LLM_Model | None = None
    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.llm is None:
            self.llm = Small_LLM_Model(model_name=self.model_name)
        self._load_vocab()

    def _load_vocab(self) -> None:
        if self.llm is None:
            raise RuntimeError("LLM instance is not initialised")
        vocab_p: str = self.llm.get_path_to_vocab_file()
        with open(vocab_p, 'r') as f:
            try:
                self.token_to_id = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise Exception(f"Vocabulary file corrupted: {e}")
        self.id_to_token = {t_id: t for t, t_id in self.token_to_id.items()}

    def get_fn_name(self, funcs: list[FuncDef],
                    prompt: str) -> Tuple[FuncDef, list[int]]:
        
        delim: int = self.llm.encode('"')[0][0]
        candidates: list[tuple[FuncDef, list[int]]] = [
            (func, self.llm.encode(func.name)[0].tolist() + [delim]) for func in funcs
        ]
        c_input_ids: list[int] = self.llm.encode(prompt)[0].tolist()
        i: int = 0
    
        while len(candidates) > 1:
            valid_t: set[int] = {token[i] for _, token in candidates}
            logits: list[float] = self.llm.get_logits_from_input_ids(c_input_ids)
            best_token = max(valid_t, key=lambda t: logits[t])
            c_input_ids.append(best_token)
            candidates = [(f, tks) for f, tks in candidates if tks[i] == best_token]
            i += 1
        
        w_func, w_tokens = candidates[0]
        c_input_ids += w_tokens[i:]
        pre_str: str = ',\n"parameters": {"'
        c_input_ids += self.llm.encode(pre_str)[0].tolist()
        return (candidates[0][0], c_input_ids)

    def get_params(self, func: FuncDef, input_ids: list[int]) -> list[int]:
        for param, p_type in func.parameters.items():
            p_tokes: list[int] = self.llm.encode(param)[0].tolist()
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)


if __name__ == "__main__":
    obj = InputParser()
    try:
        obj.parse_prompts(sys.argv[2])
        obj.parse_defs(sys.argv[1])
    except (OSError, ValueError) as e:
        print(e)
        exit()
    ls = obj.format_input()
    dcl = Decoder()
    name = dcl.get_fn_name(obj.functions, ls[0])
    print(name[0].name)
    print(dcl.llm.decode(name[1]))
