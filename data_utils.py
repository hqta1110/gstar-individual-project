import torch
import datasets
import random

def get_math_docs(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    from datasets import load_dataset
    # Replace with your actual dataset name and split if different
    ds = load_dataset("notbadai/math_reasoning", split='train')

    if eval_mode:
        # Combine fields into a single text entry for each item
        combined_texts = [
            f"Prompt: {item['prompt']}\nReasoning: {item['reasoning']}\nAnswer: {item['answer']}"
            # f"Question: {item['query']}\nResponse: {item['response']}"
            for item in ds
        ]
        
        # Limit the number of documents to a reasonable amount for profiling
        max_docs = 2000 # You can adjust this number
        sample_texts = combined_texts[:min(max_docs, len(combined_texts))]
        
        full_text = "\n\n---\n\n".join(sample_texts)
        enc = tokenizer(full_text, return_tensors="pt")
        return enc
    # The 'else' part can be left as pass if you only use eval_mode
    pass

def get_medical_docs(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    # Load the dataset from Hugging Face
    ds = datasets.load_dataset("123rc/medical_text", split='test')
    if eval_mode:
        # For evaluation, concatenate all text fields into a single string
        full_text = "\n\n".join(ds['medical_abstract'])
        enc = tokenizer(full_text, return_tensors="pt")
        return enc
    else:
        # For training sampling (not used in the main script but good practice)
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                # Select a random document
                idx = random.randint(0, len(ds) - 1)
                enc = tokenizer(ds[idx]['text'], return_tensors="pt")
                # Ensure the document is long enough for the desired sequence length
                if enc.input_ids.shape[1] >= seqlen:
                    break
            
            # Select a random chunk from the document
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = enc.input_ids[:, start:end]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    
def get_legal_docs(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    # Load the dataset from Hugging Face
    ds = datasets.load_dataset("Shekswess/legal-documents", split='train')
    if eval_mode:
        # For evaluation, concatenate all text fields into a single string
        full_text = "\n\n".join(ds['text'])
        enc = tokenizer(full_text, return_tensors="pt")
        return enc
    else:
        # For training sampling (not used in the main script but good practice)
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                # Select a random document
                idx = random.randint(0, len(ds) - 1)
                enc = tokenizer(ds[idx]['text'], return_tensors="pt")
                # Ensure the document is long enough for the desired sequence length
                if enc.input_ids.shape[1] >= seqlen:
                    break
            
            # Select a random chunk from the document
            start = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            end = start + seqlen
            inp = enc.input_ids[:, start:end]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    
def get_wikitext2(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        # testdata = datasets.load_from_disk('Your DateSets Path')['test']
        # testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
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
def get_c4_multilingual(nsamples, tokenizer, seed, seqlen, lang="en", eval_mode=False):
    """
    Load C4 dataset for a given language.
    - lang: language code, e.g., "en", "es", "fr", "de", "hi", "zh" (depends on availability in HF datasets).
    - eval_mode: if True, load validation set; else training set.
    """
    if eval_mode:
        # Load validation data for given language
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'multilingual/c4-af-validation.tfrecord-00000-of-00001.json.gz'}, split='validation')


        valenc = []
        random.seed(0)
        for _ in range(512):
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

        return TokenizerWrapper(valenc)

def get_c4(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        # valenc = valenc.input_ids[:, :(256 * seqlen)]
        # class TokenizerWrapper:
        #     def __init__(self, input_ids):
        #         self.input_ids = input_ids
        # valenc = TokenizerWrapper(valenc)
        valenc = []
        # valdata = datasets.load_from_disk('Your DateSets Path')
        random.seed(0)
        for _ in range(512):
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

def get_tinystories(nsamples, tokenizer, seed, seqlen, eval_mode=False):
    import datasets, random, torch
    
    ds_name = "roneneldan/TinyStories"
    split = "validation"
    ds = datasets.load_dataset(ds_name, split=split)

    # Choose the right field
    if "text" in ds.column_names:
        field = "text"
    elif "story" in ds.column_names:
        field = "story"
    else:
        raise ValueError(f"TinyStories dataset missing expected 'text' or 'story' field")

    if eval_mode:
        joined = "\n\n".join(ds[field])
        enc = tokenizer(joined, return_tensors="pt")
        return enc

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(ds) - 1)
            enc = tokenizer(ds[idx][field], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        l = enc.input_ids.shape[1]
        start = random.randint(0, l - seqlen - 1)
        end = start + seqlen
        inp = enc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

    
def get_loaders(name, tokenizer=None, nsamples=128, seed=0, seqlen=2048, eval_mode=False):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif 'c4' in name:
        return get_c4(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif 'tinystories' in name:
        return get_tinystories(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif "legal" in name:
        return get_legal_docs(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif 'medical' in name:
        return get_medical_docs(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif "math" in name:
        return get_math_docs(nsamples, tokenizer, seed, seqlen, eval_mode)
    elif "multilingual" in name:
        lang = name.split("-")[-1]
        return get_c4_multilingual(nsamples, tokenizer, seed, seqlen, lang, eval_mode)
    else:
        raise NotImplementedError



