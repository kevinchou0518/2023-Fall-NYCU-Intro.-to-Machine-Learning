import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip, Resize, CenterCrop
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor
import torch
from datasets import load_dataset
import json
import peft
from peft import LoraConfig, get_peft_model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = load_dataset('imagefolder', data_dir='./data/train', split='train')
dataset = dataset.train_test_split(test_size=0.05)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


checkpoint = 'microsoft/beit-large-patch16-384'
#checkpoint = "microsoft/beit-base-patch16-224-pt22k-ft22k"
#checkpoint = "microsoft/swin-base-patch4-window12-384-in22k"
#checkpoint = "microsoft/swin-base-patch4-window7-224-in22k"
#checkpoint = "microsoft/swinv2-large-patch4-window12-192-22k"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)


train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        #Resize(image_processor.size["height"]),
        #CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    del example_batch["image"]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    del example_batch["image"]
    return example_batch


train_ds = dataset["train"]
val_ds = dataset["test"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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
print_trainable_parameters(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)


lora_model.to(device)
training_args = TrainingArguments(
    output_dir="train_beit_large16_384_lora_last",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=2,
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


lora_model.save_pretrained('model_beit_large16_384_lora_last')

log_history = trainer.state.log_history
with open('./model_beit_large16_384_lora_last/log_history.json', 'w') as file:
    json.dump(log_history, file, indent=4)



