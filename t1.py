from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

dataset = load_dataset("sst2")

# Few-shot sampling
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"].select(range(100))

model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    labels=["negative", "positive"],
)

args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"sentence":"text", "label":"label"},
)

trainer.train()
metrics = trainer.evaluate(eval_dataset)
print("Test accuracy:", metrics["accuracy"])
