import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any


class SuperExpertLora:
    """
    Apply LoRA to experts, attention projections (Q/K/V), or custom targets.
    """
    def __init__(self, model: nn.Module, target_info: List[Dict[str, Any]], **lora_kwargs):
        self.base_model = model
        self.target_info = target_info

        target_modules = self._find_target_modules()
        if not target_modules:
            raise ValueError("No matching target modules found.")
        print("Targeting modules:", target_modules)

        peft_config = LoraConfig(target_modules=target_modules, **lora_kwargs)
        self.peft_model = get_peft_model(self.base_model, peft_config)

    def _find_target_modules(self) -> List[str]:
        target_module_names = []
        n_layers = len([m for n, m in self.base_model.named_modules() if n.startswith("model.layers.") and n.count(".") == 2])

        for spec in self.target_info:
            ttype = spec["type"]

            if ttype == "expert":
                base_path = f"model.layers.{spec['layer_index']}.mlp.experts.{spec['expert_index']}"
                for name, module in self.base_model.named_modules():
                    if name.startswith(base_path) and isinstance(module, nn.Linear):
                        target_module_names.append(name)

            elif ttype == "attention":
                # projections = spec.get("projections", ["q_proj", "k_proj", "v_proj"])
                projections = ['q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'] # This is for DeepSeekV2 only
                # projections = ['q_proj']
                layer_index = spec.get("layer_index", "all")

                if layer_index == "all":
                    # apply to every transformer block
                    for li in range(n_layers):
                        for proj in projections:
                            path = f"model.layers.{li}.self_attn.{proj}"
                            for name, module in self.base_model.named_modules():
                                if name == path and isinstance(module, nn.Linear):
                                    target_module_names.append(name)
                else:
                    # just one layer
                    for proj in projections:
                        path = f"model.layers.{layer_index}.self_attn.{proj}"
                        for name, module in self.base_model.named_modules():
                            if name == path and isinstance(module, nn.Linear):
                                target_module_names.append(name)

            elif ttype == "custom":
                pattern = spec["pattern"]
                for name, module in self.base_model.named_modules():
                    if pattern in name and isinstance(module, nn.Linear):
                        target_module_names.append(name)

        return sorted(set(target_module_names))

    def get_model(self) -> nn.Module:
        return self.peft_model
