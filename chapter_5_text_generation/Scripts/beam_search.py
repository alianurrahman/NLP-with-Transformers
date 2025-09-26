import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'openai-community/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

max_length = 20
input_txt = "Shiina Mahiru is"
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
output_beam = model.generate(input_ids, max_length=max_length, num_beams=5, do_sample=False)


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
    logp = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
    logp_beam = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))

    print(f"greedy: {tokenizer.decode(output_greedy[0])}")
    print(f"\nlog-prob: {logp:.2f}")

    print(f"beam: {tokenizer.decode(output_beam[0])}")
    print(f"\nlog-prob: {logp_beam:.2f}")
