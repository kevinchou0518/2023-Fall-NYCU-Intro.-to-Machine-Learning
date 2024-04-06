# Import necessary libraries
import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch
from datasets import load_dataset
import json
dataset = load_dataset('imagefolder', data_dir='./data/train', split='train')
dataset = dataset.train_test_split(test_size=0.05)
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
#checkpoint = "microsoft/swinv2-base-patch4-window12-192-22k"
#checkpoint = "microsoft/swin-base-patch4-window12-384-in22k"
#checkpoint = "microsoft/swin-base-patch4-window7-224-in22k"
#checkpoint = "microsoft/swinv2-large-patch4-window12-192-22k"
#checkpoint = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
checkpoint = "microsoft/swin-large-patch4-window12-384-in22k"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
#_transforms = Compose([Resize(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

dataset = dataset.with_transform(transforms)
data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes = True
)
model.to(device)
training_args = TrainingArguments(
    output_dir="train_swin_large4_384_22k_25",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,
    num_train_epochs=25,
    warmup_ratio=0.05,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.save_model("./model_swin_large4_384_22k_25")


log_history = trainer.state.log_history
with open('./model_swin_large4_384_22k_25/log_history.json', 'w') as file:
    json.dump(log_history, file, indent=4)



