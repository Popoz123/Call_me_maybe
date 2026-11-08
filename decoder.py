import json
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

    def get