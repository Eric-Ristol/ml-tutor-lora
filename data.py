#Everything related to the dataset: defining the Q&A pairs, formatting them
#into the SmolLM2 chat template, and building a HuggingFace Dataset object.
#We fine-tune on ML education questions so the model gets noticeably better at
#explaining machine learning concepts clearly and concisely.
import os
import json
from datasets import Dataset


#Where to save the JSON version of the dataset.
DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "qa_dataset.json")

#System prompt that sets the personality of the fine-tuned model.
SYSTEM_PROMPT = (
    "You are a helpful AI tutor specializing in machine learning. "
    "Explain concepts clearly and concisely for students."
)

#The Q&A pairs we fine-tune on. Each dict has a 'question' and an 'answer'.
#40 pairs is enough to demonstrate that LoRA fine-tuning works without making
#you wait all day. A real project would use thousands of examples.
QA_PAIRS = [
    {
        "question": "What is machine learning?",
        "answer": (
            "Machine learning is a branch of AI where we train computer programs "
            "to learn patterns from data instead of writing explicit rules. "
            "The program improves its performance on a task as it sees more examples."
        ),
    },
    {
        "question": "What is overfitting?",
        "answer": (
            "Overfitting happens when a model learns the training data too well -- "
            "including its noise and quirks -- so it performs great on training data "
            "but poorly on new, unseen data. It is like memorizing exam answers "
            "instead of understanding the concepts."
        ),
    },
    {
        "question": "What is underfitting?",
        "answer": (
            "Underfitting happens when a model is too simple to capture the real "
            "patterns in the data. It performs poorly on both training and test data. "
            "Common causes: too few parameters, too few training epochs, or too "
            "strong regularization."
        ),
    },
    {
        "question": "What is a gradient?",
        "answer": (
            "A gradient is a vector that points in the direction of the steepest "
            "increase of a function. In machine learning we compute the gradient of "
            "the loss function with respect to the model parameters, then move the "
            "parameters in the opposite direction (downhill) to reduce the loss."
        ),
    },
    {
        "question": "What is gradient descent?",
        "answer": (
            "Gradient descent is an optimization algorithm that iteratively updates "
            "model parameters by moving them in the direction opposite to the gradient "
            "of the loss function. Each step nudges the parameters toward values "
            "that reduce the loss."
        ),
    },
    {
        "question": "What is a learning rate?",
        "answer": (
            "The learning rate is a small number (like 0.001) that controls how big "
            "each gradient descent step is. Too large: the model overshoots and never "
            "converges. Too small: training is painfully slow. Finding a good learning "
            "rate is part of the art of ML."
        ),
    },
    {
        "question": "What is a loss function?",
        "answer": (
            "A loss function measures how wrong the model's predictions are. During "
            "training we try to minimize this number. Common examples: Mean Squared "
            "Error for regression, Cross-Entropy Loss for classification."
        ),
    },
    {
        "question": "What is cross-validation?",
        "answer": (
            "Cross-validation estimates how well a model generalizes. In k-fold "
            "cross-validation we split the data into k equal parts, train on k-1 "
            "parts, test on the remaining part, and repeat k times. The final score "
            "is the average across all k runs."
        ),
    },
    {
        "question": "What is regularization?",
        "answer": (
            "Regularization discourages a model from becoming too complex, reducing "
            "overfitting. The two most common types are L1 (adds the sum of absolute "
            "weights to the loss) and L2 (adds the sum of squared weights)."
        ),
    },
    {
        "question": "What is a neural network?",
        "answer": (
            "A neural network is a model loosely inspired by the brain. It consists "
            "of layers of nodes called neurons. Each neuron applies a weighted sum "
            "plus an activation function to its inputs. By stacking many layers, "
            "the network can learn very complex patterns."
        ),
    },
    {
        "question": "What is an activation function?",
        "answer": (
            "An activation function introduces non-linearity into a neural network. "
            "Without it, stacking many layers would still produce just a linear "
            "function. Common choices: ReLU (max(0,x)), Sigmoid (maps to 0-1), "
            "Tanh (maps to -1 to 1)."
        ),
    },
    {
        "question": "What is backpropagation?",
        "answer": (
            "Backpropagation computes gradients in a neural network by applying the "
            "chain rule from the output layer back to the input layer. These gradients "
            "tell us how much each weight contributed to the error, so we know "
            "exactly how to adjust them."
        ),
    },
    {
        "question": "What is a transformer?",
        "answer": (
            "A transformer is a neural network architecture that uses self-attention "
            "mechanisms to process sequences. It allows each token to attend to every "
            "other token in the sequence. Transformers are the backbone of modern "
            "language models like GPT, BERT, and SmolLM2."
        ),
    },
    {
        "question": "What is self-attention?",
        "answer": (
            "Self-attention lets each token in a sequence weigh the relevance of "
            "every other token when computing its representation. This captures "
            "long-range dependencies. The computation uses three matrices: "
            "Query, Key, and Value."
        ),
    },
    {
        "question": "What is a large language model?",
        "answer": (
            "A large language model (LLM) is a neural network trained on vast amounts "
            "of text to predict the next token in a sequence. With enough parameters "
            "and data, LLMs learn to generate fluent text, answer questions, write "
            "code, and reason about many topics."
        ),
    },
    {
        "question": "What is fine-tuning?",
        "answer": (
            "Fine-tuning is taking a pre-trained model and continuing to train it on "
            "a smaller, task-specific dataset. This adapts the model's knowledge to "
            "a new domain without training from scratch, which would require enormous "
            "compute and data."
        ),
    },
    {
        "question": "What is LoRA?",
        "answer": (
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. "
            "Instead of updating all model weights, it inserts small trainable matrices "
            "into the attention layers. This requires far fewer trainable parameters, "
            "making fine-tuning feasible on a laptop."
        ),
    },
    {
        "question": "What is transfer learning?",
        "answer": (
            "Transfer learning reuses a model trained on one task as the starting "
            "point for a different but related task. The model has already learned "
            "useful representations that transfer well, so you need much less data "
            "and compute for the new task."
        ),
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "answer": (
            "In supervised learning every training example has a label. The model "
            "learns to map inputs to outputs. In unsupervised learning there are no "
            "labels -- the model must find structure in the data on its own, for "
            "example by clustering similar examples."
        ),
    },
    {
        "question": "What is a convolutional neural network?",
        "answer": (
            "A CNN is a neural network designed for grid-structured data like images. "
            "It uses convolutional filters that slide across the input and detect local "
            "patterns such as edges and textures. This gives CNNs translation "
            "equivariance and far fewer parameters than fully connected networks."
        ),
    },
    {
        "question": "What is batch normalization?",
        "answer": (
            "Batch normalization normalizes the activations within each mini-batch to "
            "have zero mean and unit variance, then applies learnable scale and shift "
            "parameters. It stabilizes and accelerates training and has a mild "
            "regularization effect."
        ),
    },
    {
        "question": "What is a dropout layer?",
        "answer": (
            "Dropout randomly sets a fraction of neuron activations to zero during "
            "each training step. This prevents neurons from co-adapting and forces "
            "the network to learn redundant representations, acting as a regularizer "
            "that reduces overfitting."
        ),
    },
    {
        "question": "What is an embedding?",
        "answer": (
            "An embedding is a dense, low-dimensional vector representation of a "
            "discrete object like a word or user ID. Embeddings are learned during "
            "training and place similar items close together in vector space, "
            "capturing semantic relationships."
        ),
    },
    {
        "question": "What is a token in NLP?",
        "answer": (
            "A token is the basic unit of text that a language model works with. "
            "It can be a word, a sub-word, or a character depending on the tokenizer. "
            "For example, the word 'tokenization' might be split into 'token' "
            "and 'ization'."
        ),
    },
    {
        "question": "What is perplexity in language models?",
        "answer": (
            "Perplexity measures how surprised a language model is by a piece of text. "
            "Lower perplexity means the model assigns high probability to the observed "
            "tokens. A perplexity of N is roughly equivalent to having N equally "
            "likely choices at every token position."
        ),
    },
    {
        "question": "What is a hyperparameter?",
        "answer": (
            "A hyperparameter is a configuration value set before training begins, "
            "as opposed to a parameter that is learned during training. Examples: "
            "learning rate, number of layers, batch size, dropout rate. Choosing good "
            "hyperparameters is a key part of ML engineering."
        ),
    },
    {
        "question": "What is precision and recall?",
        "answer": (
            "Precision is the fraction of positive predictions that are actually "
            "positive -- it measures quality. Recall is the fraction of actual "
            "positives that were correctly identified -- it measures coverage. "
            "High precision means few false positives; high recall means few false "
            "negatives."
        ),
    },
    {
        "question": "What is the F1 score?",
        "answer": (
            "The F1 score is the harmonic mean of precision and recall: "
            "2 * (precision * recall) / (precision + recall). It balances both "
            "metrics in a single number and is especially useful when classes "
            "are imbalanced."
        ),
    },
    {
        "question": "What is a confusion matrix?",
        "answer": (
            "A confusion matrix is a table showing how often each class was predicted "
            "as each other class. Rows are true labels, columns are predicted labels. "
            "The diagonal shows correct predictions; off-diagonal cells show mistakes."
        ),
    },
    {
        "question": "What is the ROC curve?",
        "answer": (
            "The ROC (Receiver Operating Characteristic) curve plots the true positive "
            "rate against the false positive rate at various classification thresholds. "
            "The area under the curve (AUC) summarizes the model's overall ability "
            "to distinguish between classes."
        ),
    },
    {
        "question": "What is k-means clustering?",
        "answer": (
            "K-means partitions data into k clusters. It alternates between assigning "
            "each point to the nearest centroid and recomputing centroids as the mean "
            "of their assigned points, until the assignments stop changing."
        ),
    },
    {
        "question": "What is principal component analysis?",
        "answer": (
            "PCA finds the directions of maximum variance in the data (principal "
            "components) and projects the data onto a lower-dimensional subspace. "
            "It is useful for visualization and for removing noise from high-dimensional "
            "datasets."
        ),
    },
    {
        "question": "What is a random forest?",
        "answer": (
            "A random forest is an ensemble of decision trees, each trained on a "
            "random subset of the data and features. The final prediction is the "
            "majority vote (classification) or average (regression) across all trees, "
            "which reduces the overfitting of individual trees."
        ),
    },
    {
        "question": "What is boosting?",
        "answer": (
            "Boosting trains models sequentially, each one trying to correct the "
            "mistakes of the previous one. XGBoost and LightGBM are popular boosting "
            "algorithms that often win tabular data competitions."
        ),
    },
    {
        "question": "What is an autoencoder?",
        "answer": (
            "An autoencoder is trained to reconstruct its own input. An encoder "
            "compresses the input to a bottleneck representation and a decoder expands "
            "it back. The bottleneck forces the network to learn a compact, meaningful "
            "representation of the data."
        ),
    },
    {
        "question": "What is reinforcement learning?",
        "answer": (
            "Reinforcement learning trains an agent to make decisions by rewarding "
            "good actions and penalizing bad ones. The agent explores an environment, "
            "receives scalar rewards, and learns a policy that maximizes long-term "
            "cumulative reward."
        ),
    },
    {
        "question": "What is data augmentation?",
        "answer": (
            "Data augmentation artificially increases training set size by applying "
            "label-preserving transformations to existing examples. For images: "
            "flipping, cropping, color jitter. For text: synonym replacement. "
            "It reduces overfitting when you have limited data."
        ),
    },
    {
        "question": "What is the vanishing gradient problem?",
        "answer": (
            "In deep networks, gradients are multiplied together as they flow backward "
            "through layers. If they are all smaller than 1, the product shrinks "
            "exponentially and early layers receive nearly zero gradient -- they stop "
            "learning. ReLU activations and skip connections (residual networks) help."
        ),
    },
    {
        "question": "What is weight initialization?",
        "answer": (
            "Weight initialization sets starting values of model parameters before "
            "training. Good initialization (like Xavier or He) ensures gradients "
            "neither vanish nor explode at the start, making convergence faster "
            "and more reliable."
        ),
    },
    {
        "question": "What is the Adam optimizer?",
        "answer": (
            "Adam (Adaptive Moment Estimation) adapts the learning rate for each "
            "parameter individually. It keeps a running average of past gradients "
            "(momentum) and past squared gradients (RMSprop). It is the most popular "
            "optimizer for deep learning."
        ),
    },
]


def save_dataset():
    #Saves the QA_PAIRS list to a JSON file so external tools can read it.
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(QA_PAIRS, f, indent=2, ensure_ascii=False)
    print("Saved", len(QA_PAIRS), "Q&A pairs to", DATA_FILE)


def load_qa_pairs():
    #Loads Q&A pairs from the JSON file.
    #Generates it first if it does not exist yet.
    if not os.path.exists(DATA_FILE):
        save_dataset()
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def format_as_chat(qa_pair, tokenizer):
    #Formats a Q&A pair into the model's expected chat template.
    #SmolLM2-Instruct uses the ChatML format, which looks like this:
    #  <|im_start|>system
    #  You are a helpful AI tutor...<|im_end|>
    #  <|im_start|>user
    #  What is overfitting?<|im_end|>
    #  <|im_start|>assistant
    #  Overfitting happens when...<|im_end|>
    #tokenizer.apply_chat_template() builds this string automatically as long as
    #we pass a list of {"role": ..., "content": ...} dicts.

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": qa_pair["question"]},
        {"role": "assistant", "content": qa_pair["answer"]},
    ]

    #tokenize=False returns the formatted string (not token IDs yet) so we can
    #inspect it and tokenize in a separate step.
    #add_generation_prompt=False because we include the answer in training --
    #we want the model to learn to produce it, not just the prompt prefix.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def build_hf_dataset(tokenizer, max_length=512, test_split=0.1, seed=42):
    #Builds a HuggingFace Dataset object ready for the Trainer.
    #Steps:

    pairs = load_qa_pairs()
    texts = [format_as_chat(p, tokenizer) for p in pairs]

    def tokenize_example(example):
        #Tokenize a single example. Truncate if it is longer than max_length.
        #The DataCollatorForLanguageModeling handles padding per batch and
        #sets labels = input_ids automatically, so we don't set them here.
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return result

    #Wrap texts in a HuggingFace Dataset so the Trainer knows how to iterate it.
    raw_dataset = Dataset.from_dict({"text": texts})

    #Split before tokenizing so train and test have no overlap.
    split = raw_dataset.train_test_split(test_size=test_split, seed=seed)
    train_dataset = split["train"].map(tokenize_example, remove_columns=["text"])
    test_dataset  = split["test"].map(tokenize_example,  remove_columns=["text"])

    return train_dataset, test_dataset


if __name__ == "__main__":
    save_dataset()
    print("\nDataset saved. Run  python train.py  to start fine-tuning.")
