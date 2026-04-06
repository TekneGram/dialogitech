# Query Output

Query: What are the weaknesses of LLMs?
Retrieval mode: full
Rewrites: LLM limitations, Weaknesses of large language models
HyDE enabled: yes
HyDE text: A significant weakness of large language models lies in their propensity for generating plausible-sounding but factually incorrect information, a phenomenon commonly termed "hallucination." This stems from their training objective, which prioritizes linguistic coherence over absolute truth, meaning they can confidently assert falsehoods. Furthermore, LLMs often exhibit biases present in their vast training datasets, leading to skewed or discriminatory outputs regarding gender, race, or socioeconomic status. Another critical limitation is their lack of genuine understanding or reasoning capabilities; they operate based on statistical patterns rather than causal comprehension. Finally, the opacity of their complex neural network architecture presents challenges for interpretability, making it difficult to trace the source of a specific erroneous output.
Raw hits: 49
Fused hits: 15

## Ranked Chunks

### Rank 1
chunk_id: fa2389fd0ec99e337d4873948bfa9c9898181632a55a188bae78f2964b2b8022
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 1 Introduction
chunk_index: 0
classification_label: introduction
classification_source: llm
rrf_score: 0.078778
retrieval_keys: hyde_vector, original_fts, original_hybrid, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: hybrid, fts, vector, vector, vector
contributing_ranks: hyde_vector=1, original_fts=3, original_hybrid=1, rewrite_1_vector=9, rewrite_2_vector=4
relevance_score: 0.03102453052997589
distance: 0.9478736519813538
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Large Language Models (LLMs) have revolutionized language modeling and achieved remarkable success in traditional NLP tasks (Brown et al., 2020). This success has spurred their application to increasingly complex real-world scenarios (Guo et al., 2025; Claude, 2024; GPT40, 2024), particularly long-context tasks such as document-level question answering (Trivedi et al., 2022), summarization (Zhong et al., 2021), multi-shot in-context learning (Li et al., 2024b), and repository-level code generation (Bogomolov et al., 2024). These tasks typically require implicit reasoning steps that retrieve and aggregate information dispersed throughout extensive contexts before generating responses, posing significant challenges for contemporary long-context language models (LCLMs).

As a means to elicit reasoning, Chain-of-Thought (CoT) prompting has demonstrated remarkable efficacy in enhancing multi-step reasoning tasks (Wei et al., 2022; Madaan and Yazdan-bakhsh, 2022; Sprague et al., 2024). However, its effectiveness in long-context scenarios remains un-

* Equal contribution.

† Primary mentor

‡ Corresponding authors.

derexplored and only a few works have analyzed the use of CoT from limited perspectives, such as context length (Modarressi et al., 2025) , task difficulty (Bai et al., 2024b) , etc. Building on previous efforts, we for the first time conduct a systematic investigation of CoT's effectiveness across a diverse set of long-context tasks of varying lengths and domains on both open-source and proprietary models of different scales.
```

### Rank 2
chunk_id: 054dc6a7ea9f4477901e2df88908ba39857a47067e56e356ad73673a8957b88f
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: Limitations
chunk_index: 0
classification_label: discussion
classification_source: deterministic
rrf_score: 0.064789
retrieval_keys: hyde_vector, original_hybrid, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: hybrid, vector, vector, vector
contributing_ranks: hyde_vector=2, original_hybrid=3, rewrite_1_vector=1, rewrite_2_vector=1
relevance_score: 0.016393441706895828
distance: 0.7518020272254944
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Our work is still limited in some aspects. Although we have conducted experiments on Llama-3.1-8B and Qwen-2.5-7B to verify the validity of our proposed framework, we lack experiments on larger models (e.g., 32B and 70B parameters) due to limited computational resources. Similarly, due to GPU memory constraints, we can only fine-tune our model on sequences up to 16K tokens in length. We believe that fine-tuning on longer sequences would lead to better performance on long-text tasks. In future work, we plan to address these limitations and further explore the effectiveness of our framework on larger models and longer sequences.
```

### Rank 3
chunk_id: 7c077de68c70a05703e3f95202e470fdf26a892b4699c92b9d0ba486465e0469
paper_id: 2025_Uchida
paper_title: Generative AI and CEFR Levels: Evaluating the Accuracy of Text Generation with ChatGPT-4o Through Textual Features
authors: Satoru Uchida
year: 2025
section_title: Results
chunk_index: 0
classification_label: results
classification_source: deterministic
rrf_score: 0.060692
retrieval_keys: original_fts, original_hybrid, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: hybrid, fts, vector, vector
contributing_ranks: original_fts=6, original_hybrid=2, rewrite_1_vector=8, rewrite_2_vector=8
relevance_score: 0.02964426949620247
distance: 0.9729687571525574
fts_score: [none]
markdown_path: marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
marker_json_path: marker/conversion_results/2025_Uchida/2025_Uchida.json
pdf_path: pdfs/2025_Uchida.pdf
text:
```text
This section presents the results of the level estimation using CVLA and topic analysis of the texts generated by ChatGPT-4o. In this generation process, word count was not specified for either zero-shot or one-shot scenarios because it is challenging to accurately count words owing to the unique tokenization method used in large language models (LLMs). The prompt specified, "Keep the length of the passage suitable for {level}," so the AI was expected to output passages of an appropriate length based on its estimation. Table 3 presents the average word count and standard deviation for each level.

As shown in the table, the word count increased with the level in both the zero-shot and one-shot learning scenarios. However, because longer texts are typically expected at higher levels, there may be room for discussion as to whether the generated length is appropriate for those levels.
```

### Rank 4
chunk_id: cc3ac7b936a2552c6a28d029c6b74998dfc2f35fc852d669dae8a555a35be236
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 3.1 Evaluation Setup
chunk_index: 0
classification_label: method
classification_source: llm
rrf_score: 0.047643
retrieval_keys: original_hybrid, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: hybrid, vector, vector
contributing_ranks: original_hybrid=5, rewrite_1_vector=2, rewrite_2_vector=2
relevance_score: 0.016129031777381897
distance: 0.8266018629074097
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Metrics To facilitate evaluation, we structure all tasks into a multiple-choice QA format to standardize the model's output format, which enables us to directly use choice accuracy as the evaluation metric. It is worth noting that the original tasks in RULER and BABILong are not presented in a multiple-choice format. Therefore, we create multiple-choice QA pairs by combining the current sample's ground truth with ground truth answers from other samples to form the candidate options. Models To comprehensively analyze the effectiveness of CoT in long context scenarios, we conduct experiments using both open- source and proprietary models of different scales and architectures, all supporting 128k context length. The tested models include LLaMA3.1-{8B,70B}-Instruct (Dubey et al., 2024), Qwen2.5-{7B,72B}-Instruct (Yang et al., 2024b), Claude-3.5-Sonnet (Claude, 2024), GPT-40 (GPT40, 2024) and GPT-40-mini (GPT40, 2024).
```

### Rank 5
chunk_id: 2e880a77096b66f308141a28539bdcd108a41e76ba2112931c9d63f1d237e024
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: Abstract
chunk_index: 0
classification_label: abstract
classification_source: deterministic
rrf_score: 0.047170
retrieval_keys: hyde_vector, original_fts, original_hybrid
retrieval_kinds: hybrid, fts, vector
contributing_ranks: hyde_vector=6, original_fts=1, original_hybrid=4
relevance_score: 0.016393441706895828
distance: 1.1785814762115479
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Recent advances in Large Language Models (LLMs) have highlighted the challenge of handling long-context tasks, where models need to reason over extensive input contexts to aggregate target information. While Chain-of-Thought (CoT) prompting has shown promise for multi-step reasoning, its effectiveness for long-context scenarios remains underexplored. Through systematic investigation across diverse tasks, we demonstrate that CoT's benefits generalize across most long-context scenarios and amplify with increasing context length. Motivated by this critical observation, we propose LONGREPS, a process-supervised framework that teaches models to generate highquality reasoning paths for enhanced longcontext performance. Our framework incorporates a self-sampling mechanism to bootstrap reasoning paths and a novel quality assessment protocol specifically designed for longcontext scenarios. Experimental results on various long-context benchmarks demonstrate the effectiveness of our approach, achieving significant improvements over outcome supervision baselines on both in-domain tasks (+13.6/+3.8 points for LLaMA/Qwen on MuSiQue) and cross-domain generalization (+9.3/+8.1 points on average across diverse QA tasks). Our code, data and trained models are made public to facilitate future research.
```

### Rank 6
chunk_id: 283cf071b1f83b76c9665db0c3d713070a9ca5ed193a095c391b89fff5b0b826
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 4.2 CoT Quality Assessment
chunk_index: 2
classification_label: method
classification_source: llm
rrf_score: 0.046183
retrieval_keys: original_hybrid, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: hybrid, vector, vector
contributing_ranks: original_hybrid=7, rewrite_1_vector=3, rewrite_2_vector=5
relevance_score: 0.01587301678955555
distance: 0.8733982443809509
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
- Intrinsic Consistency After ensuring source faithfulness to the input text, we further examine the quality of CoT itself, which we term as Intrinsic Consistency. Specifically, a high-quality CoT should demonstrate logical coherence (appropriate question breakdown, logical use of information, and sound reasoning chain), completeness (primary reliance on retrieved information rather than model's internal knowledge), and conciseness (avoiding irrelevant or excessive details). Given the complexity of evaluating these dimensions, we employ LLM scoring with prompts detailed in Appendix A.

Finally, for each retained example, we select the CoT with the highest intrinsic consistency score for inclusion in the training data. In this way, we construct a high-quality self-sampled dataset for further model fine-tuning.
```

### Rank 7
chunk_id: df2d10a7c6cc3f9a4b4cfccaaa1df136abe509be00fa4cc64e86e97ab9a8fc06
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 7 Conclusion
chunk_index: 0
classification_label: discussion
classification_source: llm
rrf_score: 0.045991
retrieval_keys: hyde_vector, original_fts, original_hybrid
retrieval_kinds: hybrid, fts, vector
contributing_ranks: hyde_vector=3, original_fts=4, original_hybrid=9
relevance_score: 0.015625
distance: 1.1220248937606812
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
In this study, we propose LONGREPS, a processsupervised framework which enhances the longcontext reasoning capabilities of LLM by guiding them to generate high-quality reasoning paths. By conducting experiments on various long-context benchmarks, we demonstrate that our framework significantly improves model performance on both in-domain and out-domain tasks, highlighting the generalization capabilities of our method. Our experiments underscore the critical role of processsupervised reasoning paths in enhancing longcontext language models, enabling them to navigate and reason over extensive textual information with greater accuracy and reliability. By reducing the reliance on human annotations and leveraging self-sampled data, our approach offers a scalable solution for enhancing LLMs' reasoning abilities.
```

### Rank 8
chunk_id: db0a604469ff09186db7a54b9dafb3cfba0a1d53985a89704434d4105043b978
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 3.2 Observations
chunk_index: 1
classification_label: results
classification_source: llm
rrf_score: 0.045688
retrieval_keys: hyde_vector, rewrite_1_vector, rewrite_2_vector
retrieval_kinds: vector, vector, vector
contributing_ranks: hyde_vector=5, rewrite_1_vector=6, rewrite_2_vector=6
relevance_score: [none]
distance: 0.9576359391212463
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
We attribute this to the fact that S-NIAH is already sufficiently simple for most LCLMs, typically achieving 100% accuracy, and introducing

CoT may actually create interference. Notably, while Sprague et al. (2024) indicates that CoT is only effective for short-context tasks involving symbolic operations and reasoning, our observations complement their findings and demonstrate the importance of CoT prompting for long-context tasks.

Further analysis explores how CoT benefits test samples of different lengths. We divide the test data into three length tiers: Short (<32k), Medium (32k-96k), and Long (>96k), with 199, 161, and 89 samples each. Table 2 presents the performance gains achieved with CoT across all large-scale models for each length interval. We observe that the Medium and Long groups generally show higher improvements with CoT compared to the Short group. This indicates that CoT provides more benefits for longer tasks (Observation 3), possibly due to the increased need for implicit reasoning steps for information retrieval in longer contexts.
```

### Rank 9
chunk_id: d29c6d97abaf6a7c482204bc219a8a7edd7518e13d4312029f62475c1a8fff0f
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 4.2 CoT Quality Assessment
chunk_index: 0
classification_label: method
classification_source: llm
rrf_score: 0.031281
retrieval_keys: original_fts, original_hybrid
retrieval_kinds: hybrid, fts
contributing_ranks: original_fts=2, original_hybrid=6
relevance_score: 0.016129031777381897
distance: [none]
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
To identify high-quality CoT for supervised training, we evaluate CoT quality along two dimensions: answer correctness and process reliability , respectively ensuring the final answer aligns with the ground truth and the reasoning paths are logically coherent, concise, and well-supported.

Answer Correctness (AC) First, we require CoT to arrive at correct answers, filtering out failed reasoning paths. Specifically, we compute the F1-score between the CoT-derived final answer and the ground truth, retaining only reasoning paths with F1-scores above a threshold \delta . This straightforward criterion ensures the basic effectiveness of the reasoning process.

Process Reliability Beyond answer correctness, we further require the reasoning process itself to be reliable. Evaluating the reliability of reasoning processes is particularly challenging in long-context scenarios, as it requires referencing extensive input text. Even when using LLMs capable of handling long inputs, ensuring assessment accuracy remains difficult and computationally expensive. To address this challenge, we decompose process reliability into two components: source faithfulness and intrinsic consistency. The former ensures reasoning faithfulness to the source text and can be efficiently measured through simple string matching with our novel design, while the latter focuses on CoT's inherent quality and can be assessed by LLMs without requiring additional context.
```

### Rank 10
chunk_id: 7c41ccee670eb2750c535dff1ebc9b0d98ca6f8a44af6a16f36460443031969b
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 2 Related Work
chunk_index: 0
classification_label: discussion
classification_source: llm
rrf_score: 0.030579
retrieval_keys: hyde_vector, rewrite_2_vector
retrieval_kinds: vector, vector
contributing_ranks: hyde_vector=8, rewrite_2_vector=3
relevance_score: [none]
distance: 0.9284424185752869
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Long-Context Language Modeling Context length is one of the most fundamental properties of language models, determining the total amount of information they can process at once. Longcontext language modeling seeks to extend this capacity, pushing the boundaries of models' context windows. Research for context window extension primarily follows two directions. The first is data-driven strategies, which focus on constructing training data that exhibits long-range dependency patterns (Gao et al., 2024; Fu et al., 2024; Chen et al., 2024a; Dubey et al., 2024; Xiong et al., 2024; Si et al., 2024; Wang et al., 2024a; Chen et al., 2024b; Bai et al., 2024a; Wu et al., 2024) . The second is architecture-driven approaches, which enhance models' potential for processing long contexts by modifying components such as position encodings (Chen et al., 2023; Zhu et al., 2023; Peng et al., 2024; Ding et al., 2024) and attention mechanisms (An et al., 2024; Jin et al., 2024) . Building upon these advances, current LCLMs have

1 In this work, we use the terms CoT and reasoning path interchangeably. They both refer to the models' reasoning thoughts including the final answer.

demonstrated remarkable performance on simple long-context tasks, such as needle-in-haystack retrieval (Kamradt, 2023; Hsieh et al., 2024; Zhu et al., 2024) .
```

### Rank 11
chunk_id: 028ede9d26e1a42bf187f1138f1d77bc840a3aeb5fe5737c67d8cc71184b7bbf
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 4.2 CoT Quality Assessment
chunk_index: 1
classification_label: method
classification_source: llm
rrf_score: 0.030310
retrieval_keys: hyde_vector, original_fts
retrieval_kinds: fts, vector
contributing_ranks: hyde_vector=7, original_fts=5
relevance_score: [none]
distance: 1.1961610317230225
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
To address this challenge, we decompose process reliability into two components: source faithfulness and intrinsic consistency. The former ensures reasoning faithfulness to the source text and can be efficiently measured through simple string matching with our novel design, while the latter focuses on CoT's inherent quality and can be assessed by LLMs without requiring additional context. This decomposition enables efficient evaluation of reasoning processes in long-context scenarios.

- Source Faithfulness Source Faithfulness requires reasoning paths to be grounded in the actual content present in the long context. This is crucial to prevent hallucination or the inclusion of irrelevant content that could contaminate the training data. By requiring models to provide relevant excerpts during sampling (Section 4.1), we can directly employ substring exact matching to verify citation faithfulness against the source text. This enables efficient assessment of CoT's information fidelity, allowing us to filter out reasoning paths that contain content inconsistent with the original text.
- Intrinsic Consistency After ensuring source faithfulness to the input text, we further examine the quality of CoT itself, which we term as Intrinsic Consistency. Specifically, a high-quality CoT should demonstrate logical coherence (appropriate question breakdown, logical use of information, and sound reasoning chain), completeness (primary reliance on retrieved information rather than model's internal knowledge), and conciseness (avoiding irrelevant or excessive details).
```

### Rank 12
chunk_id: 80664699d50f7580e6b0990e335c91d5e39219dcb7a395b957045a751159b392
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 3.2 Observations
chunk_index: 0
classification_label: results
classification_source: llm
rrf_score: 0.030118
retrieval_keys: hyde_vector, rewrite_2_vector
retrieval_kinds: vector, vector
contributing_ranks: hyde_vector=4, rewrite_2_vector=9
relevance_score: [none]
distance: 0.9902366399765015
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
Figure 2 presents inference-time performance gain of CoT for both synthetic and real-world long context scenarios across all models.

First, we observe that CoT particularly benefits proprietary and large-scale open-source models (Observation 1) , as evidenced by the improvement on LLaMA3.1-70B-Instruct, Qwen2.5-70B-Instruct, Claude-3.5-Sonnet, GPT-4o-mini. By contrast, small scale models including LLaMA3.1-8B-Instruct and Qwen2.5-7B-Instruct fail to yield consistent performance gains from CoT. We hypothesize that this is due to smaller models' weaker overall capabilities, which may prevent them from generating high-quality reasoning paths to derive correct answers.

In addition, we find that CoT is effective across most long context scenarios, except for extremely easy retrieval tasks (Observation 2). This is consolidated by the results that most models yield improvement on complex real-world scenarios (SQA, MQA, LICL) and synthetic scenarios that requires multi-hop reasoning over long context (MNR3), while showing no gain and even suffering loss on the easiest single needle retrieval task (SNIAH). We attribute this to the fact that S-NIAH is already sufficiently simple for most LCLMs, typically achieving 100% accuracy, and introducing

CoT may actually create interference. Notably, while Sprague et al. (2024) indicates that CoT is only effective for short-context tasks involving symbolic operations and reasoning, our observations complement their findings and demonstrate the importance of CoT prompting for long-context tasks.
```

### Rank 13
chunk_id: ba6c77cb79b1da625831be70a381ab8aaf961b851a542cfd80043a45831645f3
paper_id: 2025_Uchida
paper_title: Generative AI and CEFR Levels: Evaluating the Accuracy of Text Generation with ChatGPT-4o Through Textual Features
authors: Satoru Uchida
year: 2025
section_title: Conclusion
chunk_index: 1
classification_label: discussion
classification_source: deterministic
rrf_score: 0.029670
retrieval_keys: original_hybrid, rewrite_1_vector
retrieval_kinds: hybrid, vector
contributing_ranks: original_hybrid=10, rewrite_1_vector=5
relevance_score: 0.015384615398943424
distance: 0.9294093251228333
fts_score: [none]
markdown_path: marker/conversion_results/2025_Uchida/2025_Uchida_filtered.md
marker_json_path: marker/conversion_results/2025_Uchida/2025_Uchida.json
pdf_path: pdfs/2025_Uchida.pdf
text:
```text
- RQ3: Does providing examples improve the results of text generation by ChatGPT-4o? A: Providing examples slightly improves alignment with CEFR levels and slightly increases the variety of generated topics but does not lead to dramatic improvements.

The findings emphasize the importance of checking readability, word difficulty, and topic relevance and being cautious when using ChatGPT to generate texts for CEFR levels. Although this study used the latest version of ChatGPT4o available at the time of writing, the model is expected to continue being updated, and the reproducibility of the results presented here cannot be guaranteed. As noted by Uchida (2024b), this is one of the limitations of research on generative AI. Additionally, because this study tested only one generative AI, the findings cannot be generalized to all generative AI models. Future directions may include more technically advanced approaches, such as fine-tuning generative AI, which could be a promising area of exploration. It is also necessary to investigate how human experts assess the CEFR levels of texts generated by ChatGPT. This could provide valuable insights into the alignment between AI-generated outputs and human judgment.
```

### Rank 14
chunk_id: 1126d056b0d742b85b18fdd917ab233be634c91a96c61f275523c844781790b2
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 5.1 Experimental Setups
chunk_index: 2
classification_label: method
classification_source: llm
rrf_score: 0.029631
retrieval_keys: original_hybrid, rewrite_1_vector
retrieval_kinds: hybrid, vector
contributing_ranks: original_hybrid=8, rewrite_1_vector=7
relevance_score: 0.015625
distance: 0.9715856313705444
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
For the warmup stage, we first select 300 examples and obtain their reasoning paths from the instructiontuned versions of LLaMA-3.1-8B and Qwen-2.5- 7B, then fine-tune their corresponding base models on these CoT examples for 20 steps, using a constant learning rate of 1e −5 and a global batch size of 32. In the self-training stage, we use the warmed-up models to generate reasoning paths for the remaining 3,000 examples with a temperature of 0.7. Specifically, we sample 30 reasoning paths per example and apply our proposed filtering strategy (Section 4.2) to construct the training set. We set the answer consistency threshold to 1.0, which filters out approximately 1,000 examples. The remaining examples are used to fine-tune the warmedup models for 2 epochs, with a constant learning rate of 5e −6 and a global batch size of 32. For the baseline method using outcome supervision, we directly fine-tune the warmed-up models on the complete dataset of 3,000 examples for 2 epochs, using identical hyperparameters. For fair comparison, we report the best performance achieved across the two epochs. All training procedures are conducted on 8 A100 GPUs using LLaMA-Factory (Zheng et al., 2024; Haosheng Zou and Zhang, 2024) .
```

### Rank 15
chunk_id: dd0a3f1c9eee55454ac1acf401b33a15b2c33d06d00d16b94e887b381188fb6a
paper_id: 2025_Zhu
paper_title: Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision
authors: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li
year: 2025
section_title: 1 Introduction
chunk_index: 1
classification_label: introduction
classification_source: llm
rrf_score: 0.029211
retrieval_keys: rewrite_1_vector, rewrite_2_vector
retrieval_kinds: vector, vector
contributing_ranks: rewrite_1_vector=10, rewrite_2_vector=7
relevance_score: [none]
distance: 0.9704098105430603
fts_score: [none]
markdown_path: marker/conversion_results/2025_Zhu/2025_Zhu_filtered.md
marker_json_path: marker/conversion_results/2025_Zhu/2025_Zhu.json
pdf_path: pdfs/2025_Zhu.pdf
text:
```text
derexplored and only a few works have analyzed the use of CoT from limited perspectives, such as context length (Modarressi et al., 2025) , task difficulty (Bai et al., 2024b) , etc. Building on previous efforts, we for the first time conduct a systematic investigation of CoT's effectiveness across a diverse set of long-context tasks of varying lengths and domains on both open-source and proprietary models of different scales. As illustrated in Figure 1( a), CoT's benefits generalize across most long-context scenarios, and amplify with increasing context length. Even more intriguingly, zero-shot majority voting results (Figure 1) on the long-context multihop reasoning task MuSiQue (Trivedi et al., 2022) demonstrate that CoT substantially outperforms pure majority voting, and exhibits a steeper improvement trajectory as sampling rounds increase, but still significantly lags behind its oracle performance. This naturally leads to the following research question (RQ): How to enhance models' capability in long-context scenarios to generate high-quality reasoning paths 1 for improved perfor mance?

We train LLaMA3.1-8B (Dubey et al., 2024) and Qwen2.5-7B (Yang et al., 2024a) on the MuSiQue dataset and comprehensively evaluate their performance across a diverse set of benchmarks, including: (1) MuSiQue, for assessing in-domain performance; (2) Selected QA tasks from Long-BenchV1 (Bai et al., 2023) , for measuring gen- eralization capabilities on both single and multidocument question answering; and (3) Selected QA tasks from LongBenchV2 (Bai et al., 2024b) , which feature longer contexts and more diverse domains.
```

## Summaries

### Summary Batch 0
chunk_ids: fa2389fd0ec99e337d4873948bfa9c9898181632a55a188bae78f2964b2b8022, 054dc6a7ea9f4477901e2df88908ba39857a47067e56e356ad73673a8957b88f, 7c077de68c70a05703e3f95202e470fdf26a892b4699c92b9d0ba486465e0469, cc3ac7b936a2552c6a28d029c6b74998dfc2f35fc852d669dae8a555a35be236, 2e880a77096b66f308141a28539bdcd108a41e76ba2112931c9d63f1d237e024
citation_labels: Zhu et al., 2025, Zhu et al., 2025, Uchida, 2025, Zhu et al., 2025, Zhu et al., 2025
The provided evidence does not detail inherent weaknesses of Large Language Models (LLMs) in a general sense, but rather discusses limitations in specific research contexts. For instance, in one study, researchers noted limitations due to computational constraints, such as lacking experiments on larger models (e.g., 32B and 70B parameters) and being restricted to fine-tuning on sequences up to 16K tokens due to GPU memory constraints (Zhu et al., 2025).

Additionally, in the context of text generation, it was noted that accurately counting words in texts generated by LLMs is challenging because of the unique tokenization method used (Uchida, 2025). Furthermore, while LLMs are capable of handling complex, long-context tasks that require implicit reasoning, the effectiveness of Chain-of-Thought (CoT) prompting in these long-context scenarios was previously underexplored (Zhu et al., 2025).

### Summary Batch 1
chunk_ids: 283cf071b1f83b76c9665db0c3d713070a9ca5ed193a095c391b89fff5b0b826, df2d10a7c6cc3f9a4b4cfccaaa1df136abe509be00fa4cc64e86e97ab9a8fc06, db0a604469ff09186db7a54b9dafb3cfba0a1d53985a89704434d4105043b978, d29c6d97abaf6a7c482204bc219a8a7edd7518e13d4312029f62475c1a8fff0f, 7c41ccee670eb2750c535dff1ebc9b0d98ca6f8a44af6a16f36460443031969b
citation_labels: Zhu et al., 2025, Zhu et al., 2025, Zhu et al., 2025, Zhu et al., 2025, Zhu et al., 2025
Evaluating the quality of Chain-of-Thought (CoT) reasoning paths in long-context language models (LCLMs) is challenging, particularly when assessing process reliability (Zhu et al., 2025). While source faithfulness can be measured efficiently, assessing intrinsic consistency—the inherent quality of the CoT—is difficult and computationally expensive, even with LLMs capable of handling long inputs (Zhu et al., 2025).

Furthermore, the effectiveness of CoT prompting is context-dependent. While some research suggests CoT is only effective for short-context tasks involving symbolic operations (Sprague et al., 2024), observations indicate that CoT provides greater benefits for Medium and Long context tasks compared to Short tasks (Zhu et al., 2025). In some cases, for simpler tasks, introducing CoT may actually create interference (Zhu et al., 2025).

### Summary Batch 2
chunk_ids: 028ede9d26e1a42bf187f1138f1d77bc840a3aeb5fe5737c67d8cc71184b7bbf, 80664699d50f7580e6b0990e335c91d5e39219dcb7a395b957045a751159b392, ba6c77cb79b1da625831be70a381ab8aaf961b851a542cfd80043a45831645f3, 1126d056b0d742b85b18fdd917ab233be634c91a96c61f275523c844781790b2, dd0a3f1c9eee55454ac1acf401b33a15b2c33d06d00d16b94e887b381188fb6a
citation_labels: Zhu et al., 2025, Zhu et al., 2025, Uchida, 2025, Zhu et al., 2025, Zhu et al., 2025
LLMs exhibit several weaknesses related to their reasoning and output quality. A key concern is the potential for hallucination or including irrelevant content, which necessitates ensuring "source faithfulness"—that reasoning paths are grounded in the actual source text (Zhu et al., 2025). Furthermore, the intrinsic quality of the Chain-of-Thought (CoT) itself must be assessed for logical coherence, completeness, and conciseness (Zhu et al., 2025).

In terms of performance, smaller-scale models may struggle to generate high-quality reasoning paths, leading to inconsistent performance gains from CoT (Zhu et al., 2025). Additionally, CoT may not be beneficial for extremely easy retrieval tasks, as these tasks are already simple enough for most Large Context Language Models (LCLMs) to achieve high accuracy, and introducing CoT might cause interference (Zhu et al., 2025).

Other limitations noted in generative AI include the difficulty in guaranteeing the reproducibility of results due to continuous model updates (Uchida, 2025). Moreover, studies focusing on specific models, such as ChatGPT-4o, mean that findings may not be generalizable to all generative AI models (Uchida, 2025).

## Synthesized Summary

Weaknesses in Large Language Models (LLMs) relate to output quality, reasoning, and practical limitations. LLMs can exhibit issues such as hallucination or including irrelevant content, requiring verification of "source faithfulness" in reasoning paths (Zhu et al., 2025). Assessing the intrinsic quality of Chain-of-Thought (CoT) reasoning is also challenging, as it must be evaluated for logical coherence, completeness, and conciseness (Zhu et al., 2025).

Regarding performance, the effectiveness of CoT prompting is context-dependent; while it can benefit medium and long-context tasks, it may cause interference for simpler tasks (Zhu et al., 2025). Furthermore, smaller models may struggle to produce high-quality reasoning paths, and evaluating the reliability of CoT in long-context models is difficult (Zhu et al., 2025).

Other noted limitations include computational constraints, such as restrictions on sequence length due to GPU memory (Zhu et al., 2025), and generalizability issues, as findings from specific models may not apply broadly (Uchida, 2025). Additionally, accurately counting words in LLM-generated text is complicated by unique tokenization methods (Uchida, 2025).
