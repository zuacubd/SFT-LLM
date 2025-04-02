#for skillspan
import random
import torch
import json
import numpy as np
import os
from datasets import Dataset
import evaluate
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoConfig
from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_dict({
        "idx": [example["idx"] for example in data],
        "tokens": [example["tokens"] for example in data],
        "tags_skill": [example["tags_skill"] for example in data]
    })


def count_bio_labels_before(dataset):
    bio_counts = {"B": 0, "I": 0, "O": 0}
    for example in dataset["tags_skill"]:
        for label in example:
            if label in bio_counts:
                bio_counts[label] += 1
    return bio_counts


#Setting up the data and models directory
genir_dir = "/mnt/scratch/users/genir"
arran_dir = genir_dir + "/arran"
dataset_dir = arran_dir + "/datasets" 
dataset_name = "skillspan_dataset"

#models_dir = arran_dir + "/models"
models_dir = genir_dir + "/models"
model_name = "llama2-7B"

preprocessed_dir = genir_dir + "/preprocessed"
outputs_dir = genir_dir + "/outputs"

#Loading dataset
train_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "train_new.json"))
dev_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "dev_new.json"))
test_dataset = load_dataset(os.path.join(dataset_dir, dataset_name, "test_new.json"))
#seqeval = evaluate.load("seqeval")
#seqeval = evaluate.load("token_classification")


bio_label_count = {
    "train": count_bio_labels_before(train_dataset),
    "dev": count_bio_labels_before(dev_dataset),
    "test": count_bio_labels_before(test_dataset)
}
print("BIO label counts before:", bio_label_count)
print("\n")


with open(os.path.join(preprocessed_dir, dataset_name, "bio_count_before_tokenization.json"), 'w') as f:
    json.dump(bio_label_count, f, indent=4)


label_map = {"O": 0, "B": 1, "I": 2}
label_map_inv = {0: "O", 1: "B", 2: "I"}

# Loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(models_dir, model_name))
tokenizer.pad_token = tokenizer.eos_token

#model_config = AutoConfig.from_pretrained(os.path.join(models_dir, model_name))
model = AutoModelForTokenClassification.from_pretrained(os.path.join(models_dir, model_name), 
        num_labels=len(label_map), id2label=label_map_inv, label2id=label_map, ignore_mismatched_sizes=True).bfloat16()

#LORA configuration
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
#model.classifier = torch.nn.Linear(model.config.hidden_size, 3)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=512
    )

    aligned_labels = []
    for i, word_labels in enumerate(examples["tags_skill"]):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_example = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_example.append(-100)
            elif word_idx != previous_word_idx:
                aligned_example.append(label_map[word_labels[word_idx]])
            else:
                aligned_example.append(label_map["I"] if word_labels[word_idx] != "O" else label_map["O"])

            previous_word_idx = word_idx

        aligned_labels.append(aligned_example)

    #return {
    #        "input_ids": tokenized_inputs["input_ids"],
    #        "labels": aligned_labels
    #}

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": aligned_labels
    }


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

def count_bio_labels_after(dataset, label_map_inv):
    bio_counts = {"B": 0, "I": 0, "O": 0}
    for example in dataset["labels"]:
        for label in example:
            if label != -100:
                label_str = label_map_inv[label]
                if label_str in bio_counts:
                    bio_counts[label_str] += 1
    return bio_counts

bio_label_count = {
    "train": count_bio_labels_after(train_dataset, label_map_inv),
    "dev": count_bio_labels_after(dev_dataset, label_map_inv),
    "test": count_bio_labels_after(test_dataset, label_map_inv)
}

with open(os.path.join(preprocessed_dir, dataset_name, "bio_count_after_tokenization.json"), 'w') as f:
    json.dump(bio_label_count, f, indent=4)

print("BIO label counts after:", bio_label_count)
print("\n")


def evaluate_one_shot_example(dev_dataset):

    one_shot_example = dev_dataset[21]


    inputs = tokenizer(one_shot_example["tokens"], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
    labels = one_shot_example["tags_skill"]

    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

    pred_labels = [label_map_inv[pred] for pred in predictions]
    true_labels = labels

    true_labels = true_labels[:len(pred_labels)]
    pred_labels = pred_labels[:len(true_labels)]


    print("\n")
    print("\n----One-Shot Example on Developement Set----\n")
    print("Example Tokens:", one_shot_example["tokens"])
    print("True Labels:", true_labels)
    print("Predicted Labels:", pred_labels)
    print("\n")

    report = classification_report(true_labels, pred_labels)
    print("\n----Classification Report for 1-Shot Example on Developement Set----\n", report)
    print("\n")
    print("\n")

def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)

  # Remove ignored index (special tokens)
  true_predictions = [
	  [label_map_inv[p] for (p, l) in zip(prediction, label) if l != -100]
	  for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
	  [label_map_inv[l] for (p, l) in zip(prediction, label) if l != -100]
	  for prediction, label in zip(predictions, labels)
  ]
  results = {
	  'accuracy': accuracy_score(true_labels, true_predictions),
	  'f1': f1_score(true_labels, true_predictions),
	  'classification_report': classification_report(true_labels, true_predictions)
  }
  return results


torch.cuda.empty_cache()

#Hyper-parameters
learning_rate=1e-4
batch_size=8
epochs=10 

print ("setting the training arguments...")
training_args = TrainingArguments(
    output_dir=os.path.join(outputs_dir, "skillspan_dataset/results"),
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    #logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    save_steps=500,
    save_total_limit=2,
    no_cuda=False,
    logging_strategy="epoch",
    #after evaluating each epochs performance
    load_best_model_at_end=True,
    #metric_for_best_model="f1",
    report_to="none"
)
print ("done.")

print ("setting the trainer ...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)
print ("done.")

print("\n----Begin Training----\n")
trainer.train()
print("\n")

#print ("evaluating one example")
#evaluate_one_shot_example(dev_dataset)
#print ("done")

print("\n----Train Evaluation----\n")
dev_results = trainer.evaluate(train_dataset)
print("Training set evaluation Results:", train_results)
print("\n")


print("\n----Developement Evaluation----\n")
dev_results = trainer.evaluate(dev_dataset)
print("Developement Set Evaluation Results:", dev_results)
print("\n")


print("\n----Test Evaluation----\n")
test_results = trainer.evaluate(test_dataset)
print("Test set results:", test_results)
print ("all done.")
