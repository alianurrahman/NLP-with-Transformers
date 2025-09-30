import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'openai-community/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

max_length = 50
input_txt = "Shiina Mahiru is a wife of Alianur Rahman"
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False)
output_temp = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.5, top_k=0)


def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label


def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_prob = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:]
        )
        seq_log_prob = torch.sum(log_prob[:, input_len:])
    return seq_log_prob.cpu().numpy()


if __name__ == '__main__':
    print(tokenizer.decode(output_temp[0]))
