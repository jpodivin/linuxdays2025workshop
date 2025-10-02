#!/usr/bin/env python

import argparse
import os
import random
import sys
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from peft import LoraConfig, AutoPeftModelForCausalLM, EvaConfig
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

import datasets
import torch
import tqdm

EXTRACTION_SYSTEM_PROMPT = """
You are a meticulous AI assistant designed for high-precision information extraction.
Your primary function is to create specific, fact-based question-and-answer pairs directly from a provided document.

Your task is to adhere to the following strict principles:

1.  **Absolute Grounding:** Every answer you provide MUST be a direct, verbatim quote from the source document. Do not summarize, paraphrase, or infer information.
2.  **Question Specificity:** Questions must be specific and target concrete details (e.g., names, numbers, locations, reasons, processes, definitions). Avoid overly broad or ambiguous questions that could have multiple interpretations. Questions should target information that is unique to the document.
3.  **Self-Contained Questions:** A question should be understandable on its own, but the answer must only be verifiable by consulting the provided document.
4.  **No External Knowledge:** You must operate exclusively on the information within the document. Do not use any prior knowledge.
5.  **Exclude Metadata:** Do not generate questions about the document's own properties, such as its author, title, filename, publication date, or revision history. Focus exclusively on the subject matter content within the text.
6.  **Output Format:** Your response will be a single JSON object containing a list of Q&A pairs. Each pair will have three keys: "question", "answer", and "question_topic".

Example of a GOOD question:
- "What specific percentage of the budget was allocated to research and development in the final quarter?"

Example of a BAD (too vague) question:
- "What does the document say about the budget?"
"""

BASE_EXTRACTION_PROMPT = """

Attempt to create as many high quality question / answer pairs from following document as possible.
Multiple variants of the same question / answer pair are allowed. As long as the meaning remains unchanged

Document:
```
{document}
```
"""

VARIATION_SYSTEM_PROMPT = """
You are an AI assistant specializing in linguistics and question reformulation. Your task is to rephrase a given question in multiple ways while strictly preserving its original intent and meaning.

Follow these rules:
1.  **Preserve Core Intent:** All generated questions must request the exact same piece of information as the original. The answer to all variants should be identical to the answer of the original question.
2.  **Maintain Specificity:** Do not make the questions more general or vague. If the original asks for a percentage, the variants must also ask for a percentage.
3.  **Vary Phrasing and Structure:** Use synonyms, change the sentence structure (e.g., active to passive), and reorder clauses to create natural-sounding alternatives.
4.  **Output Format:** Your response will be a single JSON object containing a list of dictionaries. Each dictionary will have only one key: "question".
"""

VARIATION_USER_PROMPT = """
Here is the original question:
{question}

Please generate {n_variants} distinct variants of this question.
"""

MODEL_DEMO_TEMPLATE = """
===================================
Query:\n{question_content}\n
Expected response:\n{answer_content}\n
Actual response:\n{decoded_response}\n
"""


class QnAPair(BaseModel):
    """Single Q/A pair with topic."""

    question: str = Field(description="Question about contents of the document.")
    answer: str = Field(description="Answer to the question")
    question_topic: str = Field(description="Topic of the question.")


class SourcedQnAPair(QnAPair):
    """Q/A pair with attached source document and topic."""

    source: str
    document_topic: str


class QnAList(BaseModel):
    """List of Q/A pairs."""

    questions_and_answers: List[QnAPair] = Field(
        description="List of questions and answers about given document.", min_length=1
    )
    document_topic: str = Field(description="Topic of the document")


class QuestionVariant(BaseModel):
    """Variant of base question"""

    question: str = Field(description="Variant of an existing question.")


class QuestionVariantList(BaseModel):
    """List of question variants."""

    question_variants: List[QuestionVariant] = Field(
        description="List of question variants.", min_length=1
    )


def get_api_key(api_key_path: str) -> str:
    """Get API key from a file"""
    with open(api_key_path, "r", encoding="utf-8") as file:
        api_key = file.read()
    return api_key


def get_device() -> str:
    """Get available torch device to train on"""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def build_dataset(
    api_key: str,
    api_url: str,
    docs_path: str,
    file_suffix: str,
    target_qna: str,
    model: str,
    parquet: str,
) -> datasets.Dataset:
    """Create dataset with documents from given dir, using LLM API of choice"""
    client = OpenAI(api_key=api_key, base_url=api_url)

    base_qna = []

    source_documents = [
        os.path.join(root, file)
        for root, _, files in os.walk(docs_path)
        for file in files
        if file.endswith(file_suffix)
    ]
    for file_path in tqdm.tqdm(source_documents, desc="Extracting Q&A from files"):
        print(f"Extracting from file {file_path}")
        with open(file_path, "r") as file:
            source_doc = file.read()
        messages = [
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": BASE_EXTRACTION_PROMPT.format(document=source_doc),
            },
        ]
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "qna-list",
                    "schema": QnAList.model_json_schema(),
                },
            },
        )
        try:
            if not isinstance(response.choices[0].message.content, str):
                raise TypeError
            parsed_response = QnAList.model_validate_json(
                response.choices[0].message.content
            )
            # Add annotations about source etc.
            for qna in parsed_response.questions_and_answers:
                base_qna.append(
                    SourcedQnAPair(
                        question=qna.question,
                        answer=qna.answer,
                        question_topic=qna.question_topic,
                        source=file_path,
                        document_topic=parsed_response.document_topic,
                    )
                )
            print(
                f"Extracted: {len(parsed_response.questions_and_answers)} QnA pairs from `{file_path}`"
            )
        except (ValidationError, TypeError) as ex:
            raise ex

    print(f"Total {len(base_qna)} extracted from {docs_path}")

    # Generate multiple variants for each question.

    final_qna = []
    for qna_pair in tqdm.tqdm(base_qna, "Generating variants"):
        messages = [
            {
                "role": "system",
                "content": VARIATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": VARIATION_USER_PROMPT.format(
                    question=qna_pair.question,
                    n_variants=target_qna,
                ),
            },
        ]
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "qna-list",
                    "schema": QuestionVariantList.model_json_schema(),
                },
            },
        )
        try:
            if not isinstance(response.choices[0].message.content, str):
                raise TypeError
            parsed_response = QuestionVariantList.model_validate_json(
                response.choices[0].message.content
            )
            # Add annotations about source etc. to new variants
            # We need to keep all the answer, and all metadata the same,
            # only crating variations of questions
            for variant in parsed_response.question_variants:
                final_qna.append(
                    SourcedQnAPair(
                        question=variant.question,
                        answer=qna_pair.answer,
                        question_topic=qna_pair.question_topic,
                        source=qna_pair.source,
                        document_topic=qna_pair.document_topic,
                    )
                )
            print(
                f"Generated: {len(parsed_response.question_variants)} QnA pairs from: '{qna_pair.question}'"
            )
        except (ValidationError, TypeError) as ex:
            raise ex
    print(f"Total {len(final_qna)} generated from {docs_path}")

    # Dicts to dataset
    dataset = []
    for e in tqdm.tqdm(final_qna, "Converting to message format"):
        # Convert to conversation format
        messages = [
            {
                "role": "user",
                "content": e.question,
            },
            {
                "role": "assistant",
                "content": e.answer,
            },
        ]
        row = e.model_dump()
        row["messages"] = messages
        dataset.append(row)

    random.shuffle(dataset)

    dataset = datasets.Dataset.from_list(dataset)

    # Save to parquet for faster retrieval
    dataset.to_parquet(parquet)

    return dataset


def create_dataset_split(
    dataset: datasets.Dataset, n_demo_samples: int = 2
) -> tuple[datasets.Dataset, datasets.Dataset, list[dict]]:
    """Create test/train/demo split for the dataset."""

    list_dataset = dataset.to_list()
    dataset_split = len(list_dataset) // 5

    demo, list_dataset = list_dataset[:n_demo_samples], list_dataset[n_demo_samples:]

    train, test = list_dataset[dataset_split:], list_dataset[:dataset_split]
    train = datasets.Dataset.from_list(train)
    test = datasets.Dataset.from_list(test)

    return train, test, demo


def get_lora_config(
    target_modules: str | list[str] = "all-linear",
    rslora: bool = True,
    dora: bool = True,
    rank: int = 16,
) -> LoraConfig:
    """Derive LoRa configuration from available HW and given params"""

    config = LoraConfig(
        r=rank,
        target_modules=target_modules,  # Fine tune only linear modules
        lora_alpha=rank,  # Set to rank for rslora
        bias="none",
        use_rslora=rslora,
        init_lora_weights="eva",  # Initialize weights base on data arxiv:2410.07170
        eva_config=EvaConfig(),
        use_dora=dora,
    )

    return config


def get_sft_config(output_dir: str, max_steps: int, device: str, finetune_name: str):
    """Configure the SFTTrainer"""

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=4,  # Set according to your device memory capacity
        learning_rate=1e-5,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=100,  # Frequency of saving model checkpoints
        eval_strategy="steps",  # Evaluate the model at regular intervals
        eval_steps=100,  # Frequency of evaluation
        use_mps_device=(device == "mps"),  # Use MPS for mixed precision training
        hub_model_id=finetune_name,  # Set a unique name for your model,
        metric_for_best_model="eval_loss",  # How we determine our improvement (used by callback)
        load_best_model_at_end=True,  # Return the last best checkpoint
        bf16=(device == "cuda"),  # Use bf16 only if you have CUDA device
        report_to="none",  # We don't want to submit reports anywhere
    )

    return sft_config


def demo_model(
    model,
    tokenizer,
    samples: list[dict],
    device: str,
) -> None:
    """Demonstrate model behavior. Each sample is a list of message dictionaries,
    alternating conversation roles.
    """

    for sample in samples:
        messages = sample["messages"]
        question = messages[0]
        answer = messages[1]
        formatted_question = tokenizer.apply_chat_template([question], tokenize=False)
        tokenized_message = tokenizer(formatted_question, return_tensors="pt").to(
            device
        )

        question_len = tokenized_message["input_ids"].shape[1]

        # Reponse should not be much longer than what we expect,
        # if it is something has gone wrong.
        # We allow for pessimistic assumption that 1 char -> 1 token.
        max_response_len = len(answer["content"]) * 2

        outputs = model.generate(
            **tokenized_message,
            max_new_tokens=max_response_len,
        )

        # Remove tokens corresponding to question message
        decoded_response = tokenizer.decode(
            outputs[0][question_len:], skip_special_tokens=True
        )

        print(
            MODEL_DEMO_TEMPLATE.format(
                question_content=question["content"],
                decoded_response=decoded_response,
                answer_content=answer["content"],
            )
        )


def train_model(
    base_model: str,
    dataset: datasets.Dataset,
    lora_config: LoraConfig,
    sft_config: SFTConfig,
    finetune_name: str,
    device: str,
    reset_chat_template: bool = True,
):
    """Fine-tune base model using dataset created."""

    train, test, demo_samples = create_dataset_split(dataset=dataset)

    # Get base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model)

    # Replace chat template with a default.
    # This will break your model, if applied without care!
    if (
        reset_chat_template
        and not hasattr(tokenizer, "chat_template")
        or tokenizer.chat_template is None
    ):
        model, tokenizer = setup_chat_format(model, tokenizer)
        print("Setting base chat template.")

    print("MODEL BEFORE TRAINING:")
    # Show how model responds before training
    demo_model(model, tokenizer, demo_samples, device=device)

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train,
        processing_class=tokenizer,
        eval_dataset=test,
        callbacks=[
            EarlyStoppingCallback(early_stopping_threshold=0.01)
        ],  # Stop if we don't improve by at least X
        peft_config=lora_config,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"./{finetune_name}")

    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=finetune_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save model in chunks of a reasonable size
    merged_model.save_pretrained(
        finetune_name, safe_serialization=True, max_shard_size="2GB"
    )

    print("MODEL AFTER TRAINING")
    # Test model after training
    demo_model(merged_model, tokenizer, demo_samples, device=device)

    return merged_model


def get_args() -> argparse.Namespace:
    """Initialize parser and get arguments."""

    parser = argparse.ArgumentParser(
        prog="infovore", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    dataset_arg_group = parser.add_argument_group(
        "dataset generation",
        description="Settings for generation of fine-tuning dataset.",
    )
    dataset_arg_group.add_argument(
        "--generate_dataset",
        action="store_true",
        help="""Generate dataset from source documents using existing model.
        This requires access to model API.""",
    )
    dataset_arg_group.add_argument(
        "--docs_path", type=str, help="Path to documentation source"
    )
    dataset_arg_group.add_argument(
        "--api_url",
        type=str,
        help="URL to LLM API. Required for dataset generation!",
    )
    dataset_arg_group.add_argument(
        "--api_key",
        type=str,
        help="Path to a file containing API key. Required for dataset generation!",
        default="./API_KEY",
    )
    dataset_arg_group.add_argument(
        "--file_suffix", type=str, default=".adoc", help="Suffix of documentation files"
    )
    dataset_arg_group.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Name of a model to request from API",
    )
    dataset_arg_group.add_argument(
        "--target_qna",
        type=int,
        default=20,
        help="Target number of QnA pairs to be created from a document.",
    )
    dataset_arg_group.add_argument(
        "--parquet",
        help="Path to save created dataset to in parquet format",
        type=str,
        default="qna.parquet",
    )

    training_arg_group = parser.add_argument_group(
        "training", description="Arguments for model fine-tuning"
    )
    training_arg_group.add_argument(
        "--base_model",
        type=str,
        help="""
        Name of a base model to fineturne, must be a valid hugging face identifier.
        https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct""",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    training_arg_group.add_argument(
        "--hf_token",
        type=str,
        help="Path to a file containing HF API key. Required for base model retrieval.",
        default="./HF_API_KEY",
    )
    training_arg_group.add_argument(
        "--model_suffix",
        type=str,
        default="NewModel",
        help="Suffix to be appended to name of base model, indicating your finetune.",
    )
    training_arg_group.add_argument(
        "--reset_chat_template",
        action="store_false",
        help="""Reset chat template if it is not set.
        This will break the model if applied without care!""",
    )
    training_arg_group.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="Path to intermediate model training checkpoints.",
    )
    training_arg_group.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of steps of the training process.",
    )
    training_arg_group.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank of LoRa matrix, higher means more parameters and more change to the base model.",
    )
    training_arg_group.add_argument(
        "--target_modules",
        nargs="+",
        type=str,
        default="all-linear",
        help="Pattern matching modules of model you want to fine tune.",
    )
    training_arg_group.add_argument(
        "--dora",
        action="store_true",
        help="""
        Use DoRa. Helps especially with low ranks. Works only on linear layers.
        https://huggingface.co/papers/2402.09353""",
    )
    training_arg_group.add_argument(
        "--rslora",
        action="store_true",
        help="""Use Rank Stabilized LoRa, improving performance.
        https://huggingface.co/papers/2312.03732""",
    )

    arguments = parser.parse_args()

    if arguments.generate_dataset:
        if not arguments.api_url:
            print("URL to model inference API is required for dataset generation!")
            sys.exit(1)
        if not arguments.api_key:
            print("Model API key is required for dataset generation!")
            sys.exit(1)

    return arguments


def main():
    args = get_args()

    # Retrieve HuggingFace API key from file
    hf_api_key = get_api_key(args.hf_token)
    # Login to HuggingFace, you need this to retrieve base model
    login(token=hf_api_key)

    # Test if dataset is already present at location if it is don't make it again.
    if not os.path.exists(args.parquet):
        if not os.path.exists(args.docs_path):
            print(f"Path to documentation source {args.docs_path} does not exist.")
            sys.exit(1)

        # Retrieve API key to generator model
        api_key = get_api_key(args.api_key)

        dataset = build_dataset(
            api_key=api_key,
            api_url=args.api_url,
            docs_path=args.docs_path,
            file_suffix=args.file_suffix,
            target_qna=args.target_qna,
            model=args.model,
            parquet=args.parquet,
        )
    else:
        dataset = datasets.Dataset.from_parquet(args.parquet)
        # Parsed parquet may end up as something else. To catch issues early,
        # and to silence linter. We are putting assertion here.
        assert isinstance(dataset, datasets.Dataset)

    # Determine what accellerator, if any, is available
    device = get_device()
    print(f"Training will use `{device}`.")

    lora_config = get_lora_config(
        target_modules=args.target_modules,
        rslora=args.rslora,
        dora=args.dora,
        rank=args.lora_rank,
    )

    model_name = args.base_model.split("/")[1]
    finetune_name = f"{model_name}-{args.model_suffix}"

    # Configure the SFTTrainer
    sft_config = get_sft_config(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=device,
        finetune_name=finetune_name,
    )

    train_model(
        args.base_model,
        dataset=dataset,
        lora_config=lora_config,
        sft_config=sft_config,
        finetune_name=finetune_name,
        device=device,
        reset_chat_template=args.reset_chat_template,
    )


if __name__ == "__main__":
    main()
