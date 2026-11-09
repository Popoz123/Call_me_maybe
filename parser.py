from typing import Literal, Annotated
import sys
from pydantic import BaseModel, ValidationError, ConfigDict, Field
import json


class ParamType(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal["string", "number", "boolean", "integer"]


class FuncDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[Annotated[str, Field(min_length=1)], ParamType]
    returns: ParamType


class PromptItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)


class InputParser(BaseModel):
    functions: list[FuncDef] = []
    prompts: list[PromptItem] = []

    @staticmethod
    def _read_f(path: str) -> list:
        with open(path, 'r') as f:
            data: list = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"File {path} must contain a JSON array")

        return data

    def parse_defs(self, path: str) -> None:
        data: list = self._read_f(path)

        for func in data:
            try:
                validated_f = FuncDef.model_validate(func)
                self.functions.append(validated_f)
            except ValidationError as e:
                raise ValueError(f"Invalid function definition {e}")

    def parse_prompts(self, path: str) -> None:
        data: list = self._read_f(path)

        for p in data:
            try:
                valid_p = PromptItem.model_validate(p)
                self.prompts.append(valid_p)
            except ValidationError as e:
                raise ValueError(f"Invalid promt {e}")

    def format_input(self) -> list[str]:
        res: list[str] = []
        p1: str = "Available functions:\n"

        for func in self.functions:
            param_l: list = []
            p1 += f"- {func.name}: {func.description} Parameters: "
            for name, param in func.parameters.items():
                param_l.append(f"{name} ({param.type})")
            p1 += f"{", ".join(param_l)}\n"

        for prompt in self.prompts:
            p2: str = f"User prompt: {prompt.prompt}\n"
            x: str = json.dumps(prompt.prompt)
            p2 += f'JSON output:{{"prompt": {x}, "name": "'
            res.append(f"{p1}\n{p2}")

        return res


if __name__ == "__main__":
    obj = InputParser()
    try:
        obj.parse_prompts(sys.argv[2])
        obj.parse_defs(sys.argv[1])
    except (OSError, ValueError) as e:
        print(e)
        exit()
    ls = obj.format_input()
    for lp in ls:
        print(lp)
