import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

subjqa = load_dataset('megagonlabs/subjqa', name='electronics')
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

model_ckpt = 'deepset/minilm-uncased-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

question = "How much music can this hold?"
context = """
An MP3 is about 1MB/minute, so about 6000 hours depending on file size."""
inputs = tokenizer(question, context, return_tensors='pt')

print(inputs['input_ids'])
print(tokenizer.decode(inputs['input_ids'][0]))


def without_pipeline():
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    print(f'Input IDs shape: {inputs.input_ids.size()}')
    print(f'Start logits shape: {start_logits.size()}')
    print(f'End logits shape: {end_logits.size()}')

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1
    answer_span = inputs['input_ids'][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_span)

    print(f'Question: {question}')
    print(f'Answer: {answer}')


def with_pipeline():
    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer)

    print(pipe(question=question, context=context, top_k=3))
    print(pipe(question='Why is there no data?', context=context, handle_imposible_answer=True))


def sliding_window_qa(context_tokens, window_size=512, stride=128):
    answers = []

    # Split context into overlapping windows
    for i in range(0, len(context_tokens), stride):
        window = context_tokens[i:i + window_size]
        inputs = tokenizer(question, window, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get answer and confidence
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        confidence = (outputs.start_logits[0, start_idx] + outputs.end_logits[0, end_idx]) / 2

        answers.append({
            'answer': tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx + 1]),
            'confidence': confidence.item(),
            'window_start': i
        })

    # Return highest confidence answer
    return max(answers, key=lambda x: x['confidence'])


def sliding_window_qa_pipeline():
    example = dfs['train'].iloc[0][['question', 'context']]
    tokenized_example = tokenizer(example['question'], example['context'], return_overflowing_tokens=True,
                                  max_length=100, stride=25)

    for idx, window in enumerate(tokenized_example['input_ids']):
        print(f'Window #{idx} has {len(window)} tokens')


if __name__ == '__main__':
    sliding_window_qa_pipeline()
