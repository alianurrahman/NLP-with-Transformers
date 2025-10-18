import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.pyplot import legend

from torch.quantization import quantize_dynamic
from pathlib import Path
from transformers import pipeline, TrainingArguments, Trainer, AutoTokenizer, AutoConfig, \
    AutoModelForSequenceClassification
from datasets import load_dataset
from evaluate import load
from time import perf_counter

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

teacher_ckpt = 'transformersbook/bert-base-uncased-finetuned-clinc'
student_ckpt = 'distilbert-base-uncased'
finetuned_ckpt = 'distillbert-base-uncased-finetuned-clinc'

# Tokenizer and dataset
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)
clinc = load_dataset('clinc_oos', 'plus')
intents = clinc['test'].features['intent']
num_labels = intents.num_classes

# Get label mappings from the teacher model
teacher_pipe = pipeline('text-classification', model=teacher_ckpt)
id2label = teacher_pipe.model.config.id2label
label2id = teacher_pipe.model.config.label2id

accuracy_score = load('accuracy')
batch_size = 48


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type='BERT baseline'):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example['text'])[0]['label']
            label = example['intent']
            preds.append(intents.str2int(pred))
            labels.append(label)
        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path('model.pt')
        torch.save(state_dict, tmp_path)

        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)

        tmp_path.unlink()
        print(f'Model size (MB) - {size_mb:.2f}')
        return {'size_mb': size_mb}

    def time_pipeline(self, query='What is the pin number for my account?'):
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.pipeline(query)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)

        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f'Average Latency (ms) - {time_avg_ms:.2f} +/- {time_std_ms:.2f}')
        return {'time_avg_ms': time_avg_ms, 'time_std_ms': time_std_ms}

    def run_benchmark(self):
        metrics = {self.optim_type: self.compute_size()}
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())

        return metrics


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_stu = model(**inputs)
        # extract cross-entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        # extract logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1)
        )
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
        return (loss, outputs_stu) if return_outputs else loss


def tokenize_text(batch):
    return student_tokenizer(batch['text'], truncation=True)


def student_init():
    return (AutoModelForSequenceClassification
            .from_pretrained(student_ckpt, config=student_config).to(device))


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


def hp_space(trial):
    return {'num_train_epochs': trial.suggest_int('num_train_epochs', 5, 10),
            'alpha': trial.suggest_float('alpha', 0, 1),
            'temperature': trial.suggest_int('temperature', 2, 20)}


clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=['text'])
clinc_enc = clinc_enc.rename_column('intent', 'labels')

student_training_args = DistillationTrainingArguments(
    output_dir=finetuned_ckpt,
    eval_strategy='epoch',
    num_train_epochs=5.0,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    alpha=1,
    weight_decay=0.01,
    push_to_hub=True
)

student_config = (AutoConfig
                  .from_pretrained(student_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id))

teacher_model = (AutoModelForSequenceClassification
                 .from_pretrained(teacher_ckpt, num_labels=num_labels).to(device))

distilbert_trainer = DistillationTrainer(model_init=student_init,
                                         teacher_model=teacher_model, args=student_training_args,
                                         train_dataset=clinc_enc['train'], eval_dataset=clinc_enc['validation'],
                                         compute_metrics=compute_metrics, tokenizer=student_tokenizer)

# best_run = distilbert_trainer.hyperparameter_search(
#     n_trial=20, direction='maximize', hp_space=hp_space()
# )
distilbert_trainer.train()
distilbert_trainer.push_to_hub('Training completed!')

# teacher benchmark
pb = PerformanceBenchmark(teacher_pipe, clinc['test'])
perf_metrics = pb.run_benchmark()

# student benchmark
student_pipe = pipeline('text-classification', model=finetuned_ckpt)
pb = PerformanceBenchmark(student_pipe, clinc['test'], optim_type='DistilBERT')
perf_metrics.update(pb.run_benchmark())

best_run = distilbert_trainer.hyperparameter_search(
    n_trials=20, direction='maximize', hp_space=hp_space
)

for k, v in best_run.hyperparameters.items():
    setattr(student_training_args, k, v)

distilled_ckpt = 'distilbert-base-uncased-distilled-clinc'
student_training_args.output_dir = distilled_ckpt

distil_trainer = DistillationTrainer(model_init=student_init,
                                     teacher_model=teacher_model, args=student_training_args,
                                     train_dataset=clinc_enc['train'], eval_dataset=clinc_enc['validation'],
                                     compute_metrics=compute_metrics(), tokenizer=student_tokenizer)
distil_trainer.train()

distilled_ckpt = 'distilbert-base-uncased-distilled-clinc'
tokenizer = AutoTokenizer.from_pretrained(distilled_ckpt)
model = (AutoModelForSequenceClassification.from_pretrained(distilled_ckpt).to('cpu'))

model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def plot_metrics(perf_metrics, current_optim_type):
    df = pd.DataFrame.from_dict(perf_metrics, orient='index')

    for idx in df.index:
        df_opt = df.loc[idx]
        if idx == current_optim_type:
            plt.scatter(df_opt['time_avg_ms'], df_opt['accuracy'] * 100, alpha=0.5, s=df_opt['size_mb'], label=idx,
                        marker='$\u25CC$')

        else:
            plt.scatter(df_opt['time_avg_ms'], df_opt['accuracy'] * 100, alpha=0.5, s=df_opt['size_mb'], label=idx)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for handle in legend.legendHandles:
        handle.set_sizes([20])

    plt.ylim(80, 90)

    xlim = int(perf_metrics['BERT baseline']['time_avg_ms'] + 3)
    plt.xlim(1, xlim)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Average latency (ms)')
    plt.show()


if __name__ == '__main__':
    plot_metrics(perf_metrics, optim_type)
