from collections import OrderedDict

import torch


class PromptCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor | None]] = OrderedDict()

    def get(self, key: tuple) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        prompt_embeds, negative_prompt_embeds = value
        return prompt_embeds.clone(), None if negative_prompt_embeds is None else negative_prompt_embeds.clone()

    def put(self, key: tuple, value: tuple[torch.Tensor, torch.Tensor | None]):
        prompt_embeds, negative_prompt_embeds = value
        self._items[key] = (
            prompt_embeds.detach().cpu(),
            None if negative_prompt_embeds is None else negative_prompt_embeds.detach().cpu(),
        )
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
