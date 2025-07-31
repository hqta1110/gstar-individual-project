import torch
import os
import psutil
process = psutil.Process(os.getpid())
from accelerate import init_empty_weights
from typing import TypeVar, Mapping
import safetensors
import json
import time

SAFETENSOR_MODEL_INDEX_FILE = "model.safetensors.index.json"

class SafeTensorsSDAdapter(Mapping[TypeVar("KT"), TypeVar("VT")]):
    def __init__(
        self,
        model_path,
        device,
        filename=SAFETENSOR_MODEL_INDEX_FILE,
        weight_map_key="weight_map",
    ):
        self.filename = filename
        model_index_file = os.path.join(model_path, self.filename)
        assert os.path.exists(model_index_file), f"file {model_index_file} not exist!"
        with open(model_index_file, "r") as f:
            self.weight_map = json.load(f)[weight_map_key]
        self.model_path = model_path
        self.device = device
        self.st_f_cache = {}
        self.total_io_time = 0

    def __getitem__(self, key):
        st = time.time()
        st_file_name = self.weight_map[key]
        f = self._get_st(st_file_name)
        v = f.get_tensor(key)
        self.total_io_time += time.time() - st
        return v

    def io_time(self):
        return self.total_io_time

    def _get_st(self, st_file_name):
        if st_file_name not in self.st_f_cache:
            sf_f_path = os.path.join(self.model_path, st_file_name)

            st_f = safetensors.safe_open(sf_f_path, framework="pt", device=self.device)
            self.st_f_cache[st_file_name] = st_f
        return self.st_f_cache[st_file_name]

    def __contains__(self, key):
        return key in self.weight_map

    def __len__(self):
        return len(self.weight_map)

    def __iter__(self):
        return iter(self.weight_map)

    def keys(self):
        return self.weight_map.keys()

    def get_shape(self, key):
        st_file_name = self.weight_map[key]
        f = self._get_st(st_file_name)
        return f.get_slice(key).get_shape()

    def get_dtype(self, key):
        st_file_name = self.weight_map[key]
        f = self._get_st(st_file_name)
        return f.get_slice(key).get_dtype()
    
def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass

def get_model(model_path):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.seqlen = 2048
    return model, tokenizer

def get_model_safe_tensors(model_path):
    model_st = SafeTensorsSDAdapter(model_path, device='cpu')
    return model_st

def print_memory_usage():
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
