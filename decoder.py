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
    number_t: set[int] = Field(default_factory=set)
    string_t: set[int] = Field(default_factory=set)

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
        if self.llm is None:
            raise RuntimeError("LLM instance is not initialised")
        vocab_p: str = self.llm.get_path_to_vocab_file()
        with open(vocab_p, 'r') as f:
            try:
                self.token_to_id = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise Exception(f"Vocabulary file corrupted: {e}")
        self.id_to_token = {t_id: t for t, t_id in self.token_to_id.items()}

        self.number_t = set()
        for t_id, t_str in self.id_to_token.items():
            if self._is_num_t(t_str):
                self.number_t.add(t_id)
        
        self.string_t = {
            t_id for t_id, t_str in self.id_to_token.items() if "\n" not in t_str
        }


    def get_fn_name(self, funcs: list[FuncDef],
                    prompt: str) -> tuple[FuncDef, list[int]]:
        
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
        pre_str: str = ',\n"parameters": {'
        c_input_ids += self.llm.encode(pre_str)[0].tolist()
        return (candidates[0][0], c_input_ids)

    def get_params(self, func: FuncDef, input_ids: list[int]) -> list[int]:
        true_t: list[int] = self.llm.encode("true")[0].tolist()
        false_t: list[int] = self.llm.encode("false")[0].tolist()
        comma_d: int = self.llm.encode(",")[0][0].item()
        brace_d: int = self.llm.encode("}")[0][0].item()
        quote_d: int = self.llm.encode('"')[0][0].item()
        brace_flag: bool = True
        
        for i, (p_name, p_type) in enumerate(func.parameters.items()):
            prefix = f'"{p_name}": ' if i == 0 else f', "{p_name}": '
            input_ids += self.llm.encode(prefix)[0].tolist()
            
            best_token: int

            if p_type.type == "boolean":
                brace_flag = True
                logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
                if logits[true_t[0]] > logits[false_t[0]]:
                    input_ids += true_t
                else:
                    input_ids += false_t

            elif p_type.type in ("number", "integer"):
                flag: bool = True
                is_last: bool = (i == len(func.parameters) - 1)
                ex_delim: int = brace_d if is_last else comma_d

                logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
                best_token = max(self.number_t, key=lambda t: logits[t])
                input_ids.append(best_token)

                while flag:
                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    best_token = max(self.number_t | {ex_delim}, key=lambda t: logits[t])
                    if best_token != ex_delim:
                        input_ids.append(best_token)
                    else:
                        if ex_delim == brace_d:
                            brace_flag = True
                        else:
                            brace_flag = False
                        flag = False
                
            
            else:
                flag: bool = True
                brace_flag = True
                input_ids.append(quote_d)

                while flag:
                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    best_token = max(self.string_t, key=lambda t: logits[t])
                    input_ids.append(best_token)
                    if best_token == quote_d:
                        break
            if brace_flag:
                input_ids.append(brace_d)

        return input_ids
                

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
    name = dcl.get_fn_name(obj.functions, ls[1])
    res = dcl.get_params(name[0], name[1])
#    print(name[0].name)
    print(dcl.llm.decode(res))
