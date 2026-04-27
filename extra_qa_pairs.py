#Additional curated ML Q&A pairs that extend the original 40 in data.py.
#Topics span classical ML, deep learning, NLP, optimization, evaluation, and
#modern LLM training techniques. Same shape as data.QA_PAIRS so we can simply
#concatenate the two lists at training time.
EXTRA_QA_PAIRS = [
    {
        "question": "What is the bias-variance tradeoff?",
        "answer": (
            "Bias is error from incorrect assumptions in the model -- a model that is "
            "too simple consistently misses the true pattern. Variance is error from "
            "sensitivity to small fluctuations in the training data. As model complexity "
            "grows, bias falls but variance rises. Good models balance the two."
        ),
    },
    {
        "question": "What is binary cross-entropy loss?",
        "answer": (
            "Binary cross-entropy measures the distance between a predicted probability "
            "and a binary label. For a true label y in {0,1} and prediction p, the loss "
            "is -[y*log(p) + (1-y)*log(1-p)]. It heavily penalizes confident wrong "
            "predictions, which makes it the standard loss for binary classifiers."
        ),
    },
    {
        "question": "What is KL divergence?",
        "answer": (
            "Kullback-Leibler divergence measures how different one probability "
            "distribution is from another. It is asymmetric -- KL(P || Q) is not the "
            "same as KL(Q || P). In ML it shows up in variational inference, knowledge "
            "distillation, and as a regularizer in policy-gradient methods."
        ),
    },
    {
        "question": "What is hinge loss?",
        "answer": (
            "Hinge loss is the loss function used by support vector machines. For a "
            "label y in {-1, +1} and score s, it is max(0, 1 - y*s). Predictions on "
            "the correct side of the margin contribute zero loss; only points inside "
            "or across the margin push the decision boundary."
        ),
    },
    {
        "question": "What is mean absolute error?",
        "answer": (
            "MAE is the average of absolute differences between predictions and true "
            "values. Compared to mean squared error it is less sensitive to outliers, "
            "since errors are not squared. The trade-off is that the gradient of MAE "
            "is constant, which can make optimization slightly less smooth."
        ),
    },
    {
        "question": "What is a support vector machine?",
        "answer": (
            "An SVM finds the hyperplane that separates two classes with the largest "
            "possible margin. Only the points closest to the boundary -- the support "
            "vectors -- determine its position. Kernels let SVMs draw non-linear "
            "boundaries by implicitly mapping inputs into a higher-dimensional space."
        ),
    },
    {
        "question": "What is a kernel in SVMs?",
        "answer": (
            "A kernel is a function that computes the similarity between two inputs as "
            "if they had been mapped into a higher-dimensional feature space, without "
            "ever computing that mapping explicitly. Common kernels are linear, "
            "polynomial, and RBF (Gaussian)."
        ),
    },
    {
        "question": "What is logistic regression?",
        "answer": (
            "Logistic regression is a linear model for binary classification. It "
            "computes a weighted sum of the inputs and squashes the result through a "
            "sigmoid to produce a probability. Despite the name, it is a classifier, "
            "not a regression model in the usual sense."
        ),
    },
    {
        "question": "What is naive Bayes?",
        "answer": (
            "Naive Bayes is a probabilistic classifier based on Bayes' theorem and a "
            "strong (naive) assumption that features are conditionally independent "
            "given the class. Despite that assumption rarely holding, it works "
            "surprisingly well on text problems and is extremely fast to train."
        ),
    },
    {
        "question": "What is a decision tree?",
        "answer": (
            "A decision tree splits the data along feature thresholds, building a "
            "tree of yes/no questions. Each leaf produces a prediction. Trees are "
            "easy to interpret but tend to overfit, which is why ensembles like random "
            "forests and gradient boosting are usually used in practice."
        ),
    },
    {
        "question": "What is k-nearest neighbors?",
        "answer": (
            "KNN classifies a new point by looking at the labels of the k closest "
            "training examples and taking a majority vote (or average for regression). "
            "It does no real training -- the training data IS the model -- so prediction "
            "can be slow on large datasets."
        ),
    },
    {
        "question": "What is one-hot encoding?",
        "answer": (
            "One-hot encoding turns a categorical variable with N levels into N binary "
            "columns, where exactly one column is 1 for each example. It avoids "
            "imposing a false ordering on unordered categories, but creates very wide, "
            "sparse feature matrices when N is large."
        ),
    },
    {
        "question": "What is feature scaling?",
        "answer": (
            "Feature scaling brings input features onto comparable ranges, typically "
            "via standardization (zero mean, unit variance) or min-max normalization "
            "(values between 0 and 1). It matters for distance- and gradient-based "
            "models, where features with large numerical ranges would otherwise "
            "dominate."
        ),
    },
    {
        "question": "What is feature engineering?",
        "answer": (
            "Feature engineering is the process of constructing input variables that "
            "make the learning task easier. Examples: log-transforming a skewed "
            "distribution, extracting day-of-week from a timestamp, building "
            "interaction terms. Good features often beat fancier models."
        ),
    },
    {
        "question": "What is class imbalance?",
        "answer": (
            "Class imbalance is when one class is far more frequent than another. A "
            "model that always predicts the majority class can score high on accuracy "
            "while being useless for the minority class. Fixes include resampling, "
            "class weights, and switching to metrics like F1 or AUC."
        ),
    },
    {
        "question": "What is SMOTE?",
        "answer": (
            "SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic "
            "examples of the minority class by interpolating between existing minority "
            "points and their nearest neighbors. It balances the dataset without simply "
            "duplicating rows, which would otherwise encourage memorization."
        ),
    },
    {
        "question": "What is bagging?",
        "answer": (
            "Bagging (bootstrap aggregating) trains many models on random sub-samples "
            "of the data, drawn with replacement, then averages their predictions. "
            "Averaging cancels out the variance of individual models. Random forests "
            "are bagging applied to decision trees."
        ),
    },
    {
        "question": "What is stacking?",
        "answer": (
            "Stacking trains several different base models, then trains a meta-model "
            "to combine their predictions. The base models capture different patterns; "
            "the meta-model learns how to weight them. It often beats any single base "
            "model on tabular benchmarks."
        ),
    },
    {
        "question": "What is XGBoost?",
        "answer": (
            "XGBoost is an efficient implementation of gradient boosting on decision "
            "trees. It adds regularization, handles missing values natively, and uses "
            "a clever second-order approximation of the loss. For years it was the "
            "default winner of tabular Kaggle competitions."
        ),
    },
    {
        "question": "What is LightGBM?",
        "answer": (
            "LightGBM is a gradient-boosting framework that grows trees leaf-wise "
            "rather than level-wise and uses histogram-based splits. The result is "
            "training speed and memory usage that are typically much better than "
            "XGBoost on large datasets, with comparable accuracy."
        ),
    },
    {
        "question": "What is early stopping?",
        "answer": (
            "Early stopping monitors validation loss during training and halts when it "
            "stops improving for a number of steps. It is a cheap, effective form of "
            "regularization: you avoid training past the point where the model begins "
            "to overfit the training data."
        ),
    },
    {
        "question": "What is weight decay?",
        "answer": (
            "Weight decay shrinks model weights toward zero on every update, which is "
            "equivalent to adding an L2 penalty to the loss. It discourages large "
            "weights, reducing the model's ability to overfit. Most modern optimizers "
            "expose it as a hyperparameter."
        ),
    },
    {
        "question": "What is grid search?",
        "answer": (
            "Grid search evaluates every combination of a discrete set of "
            "hyperparameter values. It is simple and exhaustive but scales badly: "
            "the number of trials grows multiplicatively in the number of "
            "hyperparameters."
        ),
    },
    {
        "question": "What is random search?",
        "answer": (
            "Random search samples hyperparameter combinations at random instead of "
            "trying every one on a grid. Empirically it finds good configurations "
            "faster than grid search when only a few hyperparameters actually matter, "
            "because it does not waste trials on the unimportant ones."
        ),
    },
    {
        "question": "What is Bayesian optimization for hyperparameters?",
        "answer": (
            "Bayesian optimization fits a surrogate model (often a Gaussian process) "
            "to the relationship between hyperparameters and validation score, then "
            "picks the next configuration to try by balancing exploration of uncertain "
            "regions with exploitation of promising ones."
        ),
    },
    {
        "question": "What is SGD?",
        "answer": (
            "Stochastic gradient descent updates parameters using the gradient computed "
            "on a single example or a small mini-batch instead of the full dataset. "
            "Each step is noisier than full-batch GD, but updates are far cheaper, and "
            "the noise can actually help escape shallow local minima."
        ),
    },
    {
        "question": "What is momentum in optimization?",
        "answer": (
            "Momentum accelerates gradient descent by accumulating an exponentially "
            "weighted average of past gradients and stepping in that direction. It "
            "dampens oscillations across steep dimensions and speeds up progress along "
            "consistent directions, like a ball rolling downhill."
        ),
    },
    {
        "question": "What is RMSprop?",
        "answer": (
            "RMSprop divides each parameter's gradient by a running average of its "
            "recent squared gradients. The effect is an adaptive per-parameter learning "
            "rate: parameters with large recent gradients get smaller steps, and "
            "parameters with small gradients get larger steps."
        ),
    },
    {
        "question": "What is AdaGrad?",
        "answer": (
            "AdaGrad adapts the learning rate for each parameter based on the sum of "
            "squared gradients seen so far. Parameters that get frequent large "
            "gradients have their effective learning rate reduced. It works well for "
            "sparse data but can shrink the learning rate too aggressively."
        ),
    },
    {
        "question": "What is a learning rate schedule?",
        "answer": (
            "A learning rate schedule changes the learning rate during training "
            "according to a recipe. Common schedules include step decay, exponential "
            "decay, and cosine annealing. They typically start higher to make fast "
            "progress and decrease over time to fine-tune the final solution."
        ),
    },
    {
        "question": "What is cosine learning rate annealing?",
        "answer": (
            "Cosine annealing follows half a cosine curve from the initial learning "
            "rate down to a small minimum across the training run. The smooth decay "
            "tends to give better final loss than abrupt step decays and has become a "
            "default in deep learning recipes."
        ),
    },
    {
        "question": "What is gradient clipping?",
        "answer": (
            "Gradient clipping caps the norm (or value) of the gradient before the "
            "optimizer step. It prevents an occasional huge gradient -- common when "
            "training RNNs or transformers -- from blowing up the weights. Typical "
            "thresholds for the global norm are between 0.5 and 5."
        ),
    },
    {
        "question": "What is the vanishing vs exploding gradient problem?",
        "answer": (
            "In deep networks, gradients are products of many partial derivatives. "
            "If those terms are mostly less than 1, the gradient shrinks toward zero "
            "and early layers stop learning -- vanishing. If they are greater than 1, "
            "the gradient grows and weights blow up -- exploding."
        ),
    },
    {
        "question": "What is a residual connection?",
        "answer": (
            "A residual connection adds the input of a layer (or block) to its output: "
            "y = F(x) + x. This makes it easy for the layer to learn the identity, "
            "lets gradients flow directly through the network, and is what made "
            "training networks hundreds of layers deep practical (ResNet)."
        ),
    },
    {
        "question": "What is layer normalization?",
        "answer": (
            "Layer normalization normalizes the activations of a single example across "
            "its features, so it does not depend on batch size. It is the default "
            "normalization in transformers, where batch sizes can be tiny and inputs "
            "have variable lengths."
        ),
    },
    {
        "question": "What is RMSNorm?",
        "answer": (
            "RMSNorm normalizes activations by their root-mean-square value rather "
            "than by mean and variance. It drops the mean-centering step of LayerNorm, "
            "which makes it slightly faster and works just as well in many transformer "
            "architectures."
        ),
    },
    {
        "question": "What is a recurrent neural network?",
        "answer": (
            "An RNN processes a sequence one step at a time, maintaining a hidden "
            "state that is updated with each input. The same weights are reused at "
            "every step. Plain RNNs struggle with long-range dependencies because of "
            "vanishing gradients."
        ),
    },
    {
        "question": "What is an LSTM?",
        "answer": (
            "A Long Short-Term Memory network is an RNN with gating mechanisms that "
            "control what information enters, leaves, and persists in a cell state. "
            "The gates let it carry information over many steps, fixing the vanishing-"
            "gradient problem of plain RNNs."
        ),
    },
    {
        "question": "What is a GRU?",
        "answer": (
            "A Gated Recurrent Unit is a streamlined LSTM with two gates instead of "
            "three and no separate cell state. It typically matches LSTM performance "
            "with fewer parameters and slightly faster training."
        ),
    },
    {
        "question": "What is positional encoding in transformers?",
        "answer": (
            "Self-attention is permutation-invariant, so without extra information the "
            "model cannot tell tokens apart by position. Positional encodings -- either "
            "fixed sinusoids or learned vectors -- are added to token embeddings to "
            "give the model a sense of order."
        ),
    },
    {
        "question": "What is rotary positional encoding (RoPE)?",
        "answer": (
            "RoPE encodes token positions by rotating the query and key vectors in "
            "two-dimensional subspaces by angles that depend on position. This lets "
            "the model generalize to longer contexts than it saw during training and "
            "makes relative-position information emerge naturally."
        ),
    },
    {
        "question": "What is multi-head attention?",
        "answer": (
            "Multi-head attention runs several self-attention computations in parallel, "
            "each with its own learned projection matrices, then concatenates the "
            "results. Each head can focus on different relationships in the sequence, "
            "such as syntactic vs semantic links."
        ),
    },
    {
        "question": "What is the difference between encoder, decoder, and encoder-decoder transformers?",
        "answer": (
            "Encoder-only models (like BERT) read the whole input at once and are "
            "good at understanding tasks. Decoder-only models (like GPT) generate "
            "text left-to-right. Encoder-decoder models (like T5) read an input with "
            "an encoder and emit an output with a decoder, which suits translation."
        ),
    },
    {
        "question": "What is BERT?",
        "answer": (
            "BERT is an encoder-only transformer pretrained with masked language "
            "modeling -- predict the missing words in a sentence -- plus a "
            "next-sentence prediction objective. The pretrained weights are then "
            "fine-tuned on downstream tasks like classification and QA."
        ),
    },
    {
        "question": "What is GPT?",
        "answer": (
            "GPT is a decoder-only transformer trained to predict the next token "
            "given the previous tokens (causal language modeling). At inference it "
            "generates text by repeatedly sampling the next token. Scaling parameters "
            "and data has produced increasingly capable GPT models."
        ),
    },
    {
        "question": "What is byte-pair encoding (BPE)?",
        "answer": (
            "BPE is a subword tokenization method. It starts with characters and "
            "iteratively merges the most frequent adjacent pair into a new token "
            "until the vocabulary reaches a target size. The result handles rare "
            "and unseen words gracefully by splitting them into known subwords."
        ),
    },
    {
        "question": "What is WordPiece tokenization?",
        "answer": (
            "WordPiece is the subword tokenizer used by BERT. Like BPE it builds a "
            "vocabulary of subword units, but the merging criterion is based on "
            "likelihood under a language model rather than raw frequency."
        ),
    },
    {
        "question": "What is the difference between greedy decoding and beam search?",
        "answer": (
            "Greedy decoding picks the highest-probability token at each step. Beam "
            "search keeps the top-k partial sequences alive at each step and expands "
            "all of them, which finds higher-probability full sequences but costs "
            "more compute. Greedy is faster; beam search produces better translations."
        ),
    },
    {
        "question": "What is temperature in language model sampling?",
        "answer": (
            "Temperature divides the model's logits before applying softmax. A "
            "temperature of 1 leaves the distribution unchanged. Lower temperatures "
            "(0.2-0.7) sharpen the distribution and produce more deterministic, "
            "focused output; higher temperatures produce more diverse, creative output."
        ),
    },
    {
        "question": "What is top-k sampling?",
        "answer": (
            "Top-k sampling restricts next-token sampling to the k most likely tokens "
            "and renormalizes their probabilities. It prevents the model from "
            "occasionally picking a very unlikely token, which would make the output "
            "incoherent."
        ),
    },
    {
        "question": "What is nucleus (top-p) sampling?",
        "answer": (
            "Top-p sampling keeps the smallest set of tokens whose cumulative "
            "probability exceeds p (e.g. 0.9), then samples from that set. The cutoff "
            "adapts to the shape of the distribution: tight when the model is "
            "confident, broad when it is uncertain."
        ),
    },
    {
        "question": "What is RLHF?",
        "answer": (
            "Reinforcement Learning from Human Feedback fine-tunes a language model "
            "against a reward model trained to mimic human preferences. The standard "
            "recipe is: supervised fine-tuning, train a reward model from preference "
            "comparisons, then optimize the policy with PPO against that reward."
        ),
    },
    {
        "question": "What is DPO?",
        "answer": (
            "Direct Preference Optimization replaces the RL stage of RLHF with a "
            "supervised loss derived from human preference pairs. It avoids training "
            "and using a separate reward model, which simplifies the pipeline and "
            "often matches or beats PPO-based RLHF."
        ),
    },
    {
        "question": "What is knowledge distillation?",
        "answer": (
            "Knowledge distillation trains a smaller student model to imitate the "
            "outputs of a larger teacher model. The student learns from the teacher's "
            "soft probability distributions, which carry more information than hard "
            "labels alone. The result is a compact model that retains much of the "
            "teacher's quality."
        ),
    },
    {
        "question": "What is model quantization?",
        "answer": (
            "Quantization stores model weights (and sometimes activations) in lower "
            "precision -- e.g. int8 or 4-bit -- instead of fp32. The model uses less "
            "memory and runs faster, with usually small accuracy loss. It is a key "
            "tool for deploying LLMs on consumer hardware."
        ),
    },
    {
        "question": "What is model pruning?",
        "answer": (
            "Pruning removes parameters that contribute little to the output -- often "
            "weights with small magnitude. The resulting model is sparser and faster, "
            "especially on hardware that can exploit sparsity. Pruning can be applied "
            "during or after training."
        ),
    },
    {
        "question": "What is mixture of experts?",
        "answer": (
            "A mixture-of-experts layer routes each token to a small subset of expert "
            "sub-networks, instead of using all parameters every step. This grows the "
            "total parameter count cheaply: the model has many more weights, but only "
            "a fraction is active per token."
        ),
    },
    {
        "question": "What is QLoRA?",
        "answer": (
            "QLoRA combines 4-bit quantization of the base model with LoRA adapters. "
            "It lets you fine-tune very large models on a single consumer GPU because "
            "the frozen base weights are tiny in memory and only the small LoRA "
            "matrices are trained in higher precision."
        ),
    },
    {
        "question": "What is prefix tuning?",
        "answer": (
            "Prefix tuning prepends a small set of trainable vectors to the keys and "
            "values at every transformer layer, while keeping the base model frozen. "
            "The prefixes act as a learned soft prompt that steers the model toward "
            "the target task."
        ),
    },
    {
        "question": "What is prompt engineering?",
        "answer": (
            "Prompt engineering is the practice of crafting inputs to a language model "
            "to elicit better outputs without retraining. Techniques include clear "
            "task instructions, example demonstrations (few-shot), explicit "
            "step-by-step reasoning, and structured output formats."
        ),
    },
    {
        "question": "What is chain-of-thought prompting?",
        "answer": (
            "Chain-of-thought prompting asks the model to write out intermediate "
            "reasoning steps before giving a final answer. On problems that need "
            "multi-step reasoning -- math, logic, code -- this typically improves "
            "accuracy compared to asking only for the final answer."
        ),
    },
    {
        "question": "What is in-context learning?",
        "answer": (
            "In-context learning is when a language model adapts to a task purely "
            "from examples included in the prompt, with no weight updates. The model "
            "infers the pattern from the demonstrations and applies it to the test "
            "input. It is one of the most striking emergent abilities of LLMs."
        ),
    },
    {
        "question": "What is retrieval-augmented generation?",
        "answer": (
            "RAG augments a language model with a retrieval step: it fetches "
            "relevant documents from a knowledge base, concatenates them into the "
            "prompt, and lets the model answer using that context. It reduces "
            "hallucination and lets you keep facts up to date without retraining."
        ),
    },
    {
        "question": "What is hallucination in language models?",
        "answer": (
            "A hallucination is content the model produces that sounds plausible but "
            "is not grounded in fact or in the provided context -- fabricated "
            "citations, made-up APIs, invented quotes. It is a fundamental risk of "
            "models trained to predict likely text rather than true text."
        ),
    },
    {
        "question": "What is BLEU?",
        "answer": (
            "BLEU is a metric for machine translation that compares the candidate "
            "translation to one or more reference translations using n-gram overlap, "
            "with a brevity penalty. It is fast and language-agnostic, but correlates "
            "imperfectly with human judgement."
        ),
    },
    {
        "question": "What is ROUGE?",
        "answer": (
            "ROUGE is a family of metrics for summarization that measures n-gram "
            "and longest-common-subsequence overlap between the generated summary and "
            "reference summaries. ROUGE-1, ROUGE-2, and ROUGE-L are the most commonly "
            "reported variants."
        ),
    },
    {
        "question": "What is model calibration?",
        "answer": (
            "A model is calibrated if its predicted probabilities match observed "
            "frequencies: among examples the model says are 70% likely positive, "
            "about 70% should actually be positive. Modern deep classifiers are "
            "often miscalibrated; temperature scaling is a simple fix."
        ),
    },
    {
        "question": "What is t-SNE?",
        "answer": (
            "t-SNE is a non-linear dimensionality-reduction technique used mostly for "
            "visualization. It places similar high-dimensional points close together "
            "in 2D or 3D. The global structure can be misleading, but local clusters "
            "are usually meaningful."
        ),
    },
    {
        "question": "What is UMAP?",
        "answer": (
            "UMAP is a non-linear dimensionality-reduction technique similar in spirit "
            "to t-SNE but typically faster and better at preserving global structure. "
            "It is the default visualization tool for many embedding analyses."
        ),
    },
    {
        "question": "What is DBSCAN?",
        "answer": (
            "DBSCAN clusters points that are densely packed together and labels "
            "points in low-density regions as noise. Unlike k-means it does not need "
            "the number of clusters in advance and can find clusters of arbitrary "
            "shape."
        ),
    },
    {
        "question": "What is hierarchical clustering?",
        "answer": (
            "Hierarchical clustering builds a tree of clusters -- either by merging "
            "the closest pairs (agglomerative) or splitting clusters (divisive). The "
            "resulting dendrogram lets you choose the number of clusters after the "
            "fact by cutting at different heights."
        ),
    },
    {
        "question": "What is collaborative filtering?",
        "answer": (
            "Collaborative filtering recommends items to a user based on the "
            "preferences of similar users. It needs only the interaction history -- "
            "ratings, clicks, purchases -- and does not rely on item content features."
        ),
    },
    {
        "question": "What is matrix factorization for recommendations?",
        "answer": (
            "Matrix factorization decomposes the user-item interaction matrix into "
            "two low-rank matrices: one of user latent factors, one of item latent "
            "factors. Predictions for unseen pairs come from dot products of the "
            "corresponding rows and columns."
        ),
    },
    {
        "question": "What is a generative adversarial network?",
        "answer": (
            "A GAN trains two networks against each other: a generator that produces "
            "fake samples and a discriminator that tries to tell real samples from "
            "fake. The generator improves until its samples become indistinguishable "
            "from real data."
        ),
    },
    {
        "question": "What is a variational autoencoder?",
        "answer": (
            "A VAE is an autoencoder whose encoder outputs a distribution over latent "
            "codes rather than a single code. Training maximizes a lower bound on the "
            "data likelihood, which combines reconstruction error with a KL term that "
            "keeps the latent distribution close to a standard Gaussian."
        ),
    },
    {
        "question": "What is a diffusion model?",
        "answer": (
            "A diffusion model learns to reverse a process that gradually adds noise "
            "to data. Generation starts from pure noise and runs the learned reverse "
            "process step by step, denoising into a sample. Diffusion is the basis of "
            "modern image and video generators."
        ),
    },
    {
        "question": "What is anomaly detection?",
        "answer": (
            "Anomaly detection identifies examples that differ significantly from the "
            "rest of the data -- fraudulent transactions, broken sensors, intrusions. "
            "Methods range from simple statistical thresholds to isolation forests and "
            "autoencoder reconstruction error."
        ),
    },
    {
        "question": "What is multi-task learning?",
        "answer": (
            "Multi-task learning trains a single model on several related tasks at "
            "once, with shared layers and task-specific heads. Sharing representations "
            "regularizes the model and often improves performance on each individual "
            "task, especially when some have little data."
        ),
    },
    {
        "question": "What is few-shot learning?",
        "answer": (
            "Few-shot learning aims to perform well on a task given only a handful of "
            "labelled examples, by leveraging knowledge from related tasks or "
            "pretraining. In-context learning with LLMs is the most visible recent "
            "example."
        ),
    },
    {
        "question": "What is zero-shot learning?",
        "answer": (
            "Zero-shot learning solves a task without any task-specific labelled "
            "examples, typically by relying on a pretrained model's general "
            "capabilities and a description of the task in natural language."
        ),
    },
    {
        "question": "What is semi-supervised learning?",
        "answer": (
            "Semi-supervised learning uses a small set of labelled examples together "
            "with a much larger set of unlabelled examples. Techniques like "
            "self-training and consistency regularization let the model extract "
            "useful structure from the unlabelled data."
        ),
    },
    {
        "question": "What is active learning?",
        "answer": (
            "Active learning lets the model choose which examples to label next, "
            "typically the ones it is most uncertain about. With a fixed labelling "
            "budget it can reach higher accuracy than random labelling, by spending "
            "the budget where it matters most."
        ),
    },
    {
        "question": "What is the curse of dimensionality?",
        "answer": (
            "As the number of features grows, the volume of the feature space grows "
            "exponentially, so data becomes sparse and distance-based methods stop "
            "working well. Many algorithms need vastly more data, more carefully "
            "engineered features, or dimensionality reduction to cope."
        ),
    },
    {
        "question": "What is data leakage?",
        "answer": (
            "Data leakage is when information from outside the training data sneaks "
            "in -- e.g. computing feature statistics on the full dataset before "
            "splitting, or using a target-derived feature. The model looks great in "
            "validation and falls apart in production. Always split first, then "
            "preprocess."
        ),
    },
]
