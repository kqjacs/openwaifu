from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import trange
import random
import torch


random.seed(2)
torch.set_grad_enabled(False)
text = open("prompt").read()
prompt, *qa = list(open("prompt"))
q, a = qa[::2], qa[1::2]
qa = list(zip(q, a))
model_name = "facebook/opt-2.7b"
# model_name = "EleutherAI/gpt-neo-2.7B"
model_name = "EleutherAI/pythia-2.8b-deduped"
# model_name = "KoboldAI/fairseq-dense-2.7B"
# model_name = "facebook/opt-6.7b"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "KoboldAI/fairseq-dense-6.7B"
# model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             low_cpu_mem_usage=True,
                                             load_in_8bit=True,
                                             device_map="auto"
                                             )
x, y = 0, 0
samples = 16
for _ in trange(samples):
    random.shuffle(qa)
    text = "<|endoftext|>" + prompt + "".join(q + a for q, a in qa[:10])
    tokens = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    logits = model(tokens).logits
    logits = logits.float()
    if abs(logits.exp().sum(-1).mean().item() - 1) > 1e-4:
        logits = torch.log_softmax(logits, -1)
    x += torch.gather(logits, -1, tokens.unsqueeze(-1)).sum().item() / samples
    y += torch.gather(logits, -1, tokens[:, len(tokenizer.encode(prompt)):].unsqueeze(-1)).sum().item() / samples
# facebook/opt-2.7b -3861.646743774414 -3654.82
# facebook/opt-6.7b -3974.6465759277344 -3742.7179565429688
# KoboldAI/fairseq-dense-2.7B -4089.2803955078125 -3938.384078979492
# KoboldAI/fairseq-dense-6.7B -4058.0118103027344 -3837.853530883789
# EleutherAI/pythia-2.8b-deduped -4156.777084350586 -3825.0567169189453
# EleutherAI/gpt-neo-2.7B -3997.294387817383 -3686.868896484375
# EleutherAI/gpt-j-6B -4114.580902099609 -3824.721878051758
print(model_name, x, y)