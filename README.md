## The evaluation of 1.5B LLM, as well as how to improve its performance and generalize to other tasks （Started on April 8, 2025）

## Task Requirment
1. Take a look at the ControlAgent work (https://arxiv.org/pdf/2410.19811). The code is here https://github.com/ControlAgent/ControlAgent
This paper develops an agent for control design. In the paper, we have not included the cases where the base LLM model is small (say something at the level of 1.5B), and the evaluation is done on a dataset we call ControlEval. Can you run ControlAgent with a small LLM at the level of 1.5B and report the performance on ControlEval? (You need to figure out from the Github repo how the evaluation is automatically done).
2. Clearly 1.5B model will not give a satisfying performance. Can you do something to improve the performance of ControlAgent with small LMs?
3. Does your method generalize to other engineering design problems? Say circuit design? (https://arxiv.org/pdf/2405.14918) Github:https://github.com/laiyao1/AnalogCoder

## My Plan
1️⃣ **For the Task 1**\
✅ Update the model invocation method by switching from the OpenAI API to local inference using Qwen-1.5B. \
✅ Implement evaluation scripts to aggregate output results for ASR performance testing.

2️⃣ **For the Task 2** \
✅ Where does the 1.5B LLM underperform? Analyze which parts lead to failure and why.  (It failed to correctly grasp fundamental concepts in automating control) \
✅ Based on that analysis, consider targeted solutions—such as fine-tuning or prompt engineering. (By synthesizing 139 domain data, a 2% improvement was achieved with 1.5B LLM, which was scaled up to 1000 in the next step.)

3️⃣ **For the Task 3** \
✅ Essentially, it’s about designing different workflows for the LLM to process data and invoke tools, based on the specific tasks.

## Key Summay
**1 Foundation Building**: Completed evaluation code and implementation of small model inference, as well as constraint-based text formatting and generation. \
**2 1.5B LLM Enhancement**: Synthesized datasets based on prompts and errors to further enhance the small model.


## Key Report 
------------------------------------------Task 1------------------------------------------ \
**2025/4/8 16:00**: Save the final results in `save_result.py` and compute ASR metric in `asr_computing.py`. \
**2025/4/8 16:52**: Replace the OpenAI API call inside the class GPT4 with a local inference of the Qwen-1.5B LLM using Transformers. However, due to the limited capabilities, it struggles to follow the predefined prompts and frequently produces errors, making it difficult to complete a full performance evaluation. such as `"parameter": [ω_L, β_b]`, `"design": " \\ comment`, `"parameter": [, ]`. \
**2025/4/8 18:42**: Used Outlines to constrain the model’s output, designed a schema, and modified parts of the parameter-passing process to ensure error-free execution. Due to the limited capabilities of the 1.5B LLM, the output needs to be tightly constrained. \
**2025/4/8 19:45**: It's very slow, It takes about an hour to run a dozen or so pieces of data, for two reasons: 1. outlines wraps a layer. 2. almost every piece of data has to be tried the maximum number of times. Reduce the maximum number of attempts. \
**2025/4/8 22:25**: Get fist order stable result in `first_order_stable_fast_data_final_result.csv`. We get ASR 0.14. Complete log in `1-5B-llm.log` \
------------------------------------------Task 2------------------------------------------ \
**2025/4/9 00:24**: Test for understanding of the basics: Chatgpt and grok determines whether LLM 1.5B has a correct understanding of knowledge related to automatic control. Asked the 1.5B model to answer questions like “What are settling time and phase margin in automatic control systems?”
ChatGPT evaluated the answers and found them incorrect. Further verification using Grok confirmed that ChatGPT’s judgment was accurate. \
**2025/4/9 00:37**: We need to distill accurate knowledge from more powerful models to help the 1.5B model learn effectively. \
**2025/4/9 14:44**: ~~Language Agent Finetuning maybe a good idea. We can have the 1.5B LLM learn from GPT-4's reasoning trajectories.~~ \
**2025/4/9 20:23**: Completed the initial version of code for fine-tuning a 1.5B parameter LLM `1-5B_learning.py`. \
**2025/4/9 22:26**: Based on the existing agent instructions, synthesize over 100 domain-specific data samples to ensure the model minimizes errors in domain-related knowledge `auto_control_datasets.jsonl`. \
**2025/4/10 02:08**: With 139 synthesized data, the effect goes from 0.14 to 0.16 (`first_order_stable_fast_data_final_result_1-5B_finetuning.csv`)and is about to expand to 1000 synthesized data. There was a language confusion problem, but it was solved by using the Lora. \
**2025/4/10 14:40**: Tried generating more datasets but no better to improve results. \
------------------------------------------Task 3------------------------------------------ \
**2025/4/10 17:34**: For circuit design, the essence is the same.

## Reference
* [Fine-Tuning Small Language Models: Practical Recommendations](https://medium.com/@liana.napalkova/fine-tuning-small-language-models-practical-recommendations-68f32b0535ca)
* [T1: Tool-integrated Self-verification for Test-time Compute Scaling in Small Language Models](https://arxiv.org/pdf/2504.04718)
* [Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs ](https://arxiv.org/abs/2412.13337)
* [Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/pdf/2502.12143)
* [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/pdf/2501.04519)
* [Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training](https://arxiv.org/abs/2501.11425)

## Originality

This work is based on the following paper:

```
@article{guo2024controlagent,
  title={ControlAgent: Automating Control System Design via Novel Integration of LLM Agents and Domain Expertise},
  author={Guo, Xingang and Keivan, Darioush and Syed, Usman and Qin, Lianhui and Zhang, Huan and Dullerud, Geir and Seiler, Peter and Hu, Bin},
  journal={arXiv preprint arXiv:2410.19811},
  year={2024}
}
```
