# LLM-Techniques-ICL-RAG-LoRA/
├── README.md
└── llm-techniques-icl-rag-lora.ipynb


# Practical LLM Techniques: In-Context Learning, RAG, and LoRA

A hands-on Jupyter Notebook demonstrating three ways to adapt a pretrained language model to a task:

1. **In-context learning (ICL)** — guide a model with examples in the prompt without changing its weights.
2. **Retrieval-augmented generation (RAG)** — retrieve relevant passages from a small document collection and use them to ground answers.
3. **Low-Rank Adaptation (LoRA)** — fine-tune a small set of adapter parameters for a simple classification task.

The notebook uses a small FLAN-T5 model and a sentence-transformer embedding model so the experiments can run on modest hardware, although runtime depends on your environment.

## Notebook contents

### 1. In-context learning
- Compare zero-shot and few-shot sentiment classification.
- Evaluate predictions on a fixed set of sample reviews.
- Check that model weights remain unchanged during inference.

### 2. Retrieval-augmented generation
- Create and split a small fictional support-handbook collection.
- Compare lexical TF-IDF retrieval with dense semantic retrieval.
- Build a FAISS vector store and a LangChain retriever.
- Construct a context-grounded prompt with an explicit abstention response.
- Compare a direct LLM answer with a RAG answer.
- Visualize dense retrieval similarity scores.

### 3. Low-Rank Adaptation (LoRA)
- Compare full fine-tuning parameter counts with LoRA parameter counts.
- Train LoRA adapters on a small support-ticket routing task.
- Compare predictions before and after adaptation on held-out examples.

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- PEFT (LoRA)
- LangChain
- FAISS CPU
- scikit-learn
- NumPy, pandas, and Matplotlib

## Models used

- `google/flan-t5-small` — text generation and instruction-following.
- `sentence-transformers/all-MiniLM-L6-v2` — dense embeddings for semantic retrieval.

The notebook pins model revisions in code for reproducibility. The first run downloads the model files, so an internet connection is needed initially.

## Getting started

### Option A: Google Colab
1. Upload `code file 1.py.ipynb` to Google Colab.
2. Select **Runtime → Run all**.
3. Choose a GPU runtime if one is available; the notebook automatically selects CUDA when available and otherwise uses CPU.
4. Run the installation cell and allow the model checkpoints to download.

**Important:** The notebook's setup instructions say not to restart the runtime after the package-installation cell.

### Option B: Run locally
1. Install Python (a recent Python 3 version is recommended).
2. Create and activate a virtual environment.
3. Start Jupyter Notebook or JupyterLab.
4. Open the notebook and run the setup cell followed by the remaining cells in order.

The notebook installs pinned versions of its main dependencies in the first code cell:

```bash
pip install \
  "transformers==4.57.3" \
  "sentence-transformers==5.1.2" \
  "peft==0.18.0" \
  "accelerate==1.12.0" \
  "sentencepiece==0.2.1" \
  "langchain-core==1.2.7" \
  "langchain-community==0.4.1" \
  "langchain-text-splitters==1.1.0" \
  "faiss-cpu==1.13.2"
```

Other libraries used in the notebook include `torch`, `numpy`, `pandas`, `matplotlib`, `matplotlib-inline`, and `scikit-learn`. Install any missing dependencies in your environment if needed.

## How to use the notebook

- Run cells **in order**, because later sections reuse models, variables, and functions defined earlier.
- Inspect the printed tables and charts to compare retrieval methods and model behavior.
- You can modify the example reviews, handbook passages, questions, and support tickets to run your own experiments.
- Keep the fictional handbook data clearly separated from real policies or operational documents.

## Important limitations

- The support handbook and ticket examples are **fictional teaching data**, not real institutional policies.
- This is an educational experiment, not a production chatbot or a benchmark of model quality.
- Small models can produce incorrect or inconsistent outputs. RAG can improve grounding, but it does not guarantee factual correctness.
- The LoRA experiment uses a tiny synthetic dataset and a small held-out test set. Its results should not be interpreted as evidence of real-world generalization.
- CPU execution is supported by the device-selection logic but may be slower, particularly during model loading and LoRA training.
- Do not use this notebook's outputs as authoritative guidance for high-stakes decisions.

## Repository structure

```text
.
├── README.md
└── code file 1.py.ipynb
```

You may want to rename the notebook to something cleaner for GitHub, such as `llm-techniques-icl-rag-lora.ipynb`, and update the filename above accordingly.

## Learning outcomes

After working through the notebook, you should be able to explain the difference between:
- prompt-based adaptation (ICL),
- retrieving external evidence at inference time (RAG), and
- updating a small number of trainable parameters (LoRA).

## License

No license is specified in the notebook. Add a license file before describing the repository as open source. Check the licenses and terms for the pretrained models and dependencies before redistributing or deploying them.

