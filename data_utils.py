import torch
import datasets
import random

def get_wikitext2(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    if eval_mode:
        # testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        testdata = datasets.load_from_disk('Your DateSets Path')['test']
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        # traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = datasets.load_from_disk('Your DateSets Path')['train']
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_c4(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    if eval_mode:
        # valdata = datasets.load_dataset(
        # 'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        # valenc = valenc.input_ids[:, :(256 * seqlen)]
        # class TokenizerWrapper:
        #     def __init__(self, input_ids):
        #         self.input_ids = input_ids
        # valenc = TokenizerWrapper(valenc)
        valenc = []
        valdata = datasets.load_from_disk('Your DateSets Path')
        random.seed(0)
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    break

            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_from_disk('Your DateSets Path')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    
def get_loaders(name, tokenizer=None, nsamples=128, seed=0, seqlen=2048, eval_mode=False):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif 'c4' in name:
        return get_c4(nsamples, tokenizer, seed, seqlen, eval_mode)
    else:
        raise NotImplementedError



