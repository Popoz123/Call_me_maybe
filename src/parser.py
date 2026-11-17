from typing import Literal, Annotated, Any
from pydantic import BaseModel, ValidationError, ConfigDict, Field, field_validator # noqa
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

    @field_validator('prompt')
    @classmethod
    def empty_p(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be blank or whitespace only")
        return v


class InputParser(BaseModel):
    functions: list[FuncDef] = []
    prompts: list[PromptItem] = []

    @staticmethod
    def _read_f(path: str) -> list[Any]:
        with open(path, 'r', encoding="utf-8") as f:
            data: list[Any] = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"File {path} must contain a JSON array")

        return data

    def parse_defs(self, path: str) -> None:
        self.functions = []
        data: list[Any] = self._read_f(path)

        for func in data:
            try:
                validated_f = FuncDef.model_validate(func)
                self.functions.append(validated_f)
            except ValidationError as e:
                raise ValueError(f"Invalid function definition {e}")

    def parse_prompts(self, path: str) -> None:
        data: list[Any] = self._read_f(path)

        for p in data:
            try:
                valid_p = PromptItem.model_validate(p)
                self.prompts.append(valid_p)
            except ValidationError as e:
                raise ValueError(f"Invalid prompt structure\n{e}")

    def format_input(self) -> list[str]:
        res: list[str] = []
        p1: str = (
            "You are a function calling agent. Provide concise, minimal "
            "arguments.\nAvailable functions:\n"
        )

        for func in self.functions:
            param_l: list[str] = []
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
