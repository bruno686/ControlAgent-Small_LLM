## The evaluation of 1.5B LLM, as well as how to improve its performance and generalize to other tasks （Started on April 8, 2025）

## Task Requirment
1. Take a look at the ControlAgent work (https://arxiv.org/pdf/2410.19811). The code is here https://github.com/ControlAgent/ControlAgent
This paper develops an agent for control design. In the paper, we have not included the cases where the base LLM model is small (say something at the level of 1.5B), and the evaluation is done on a dataset we call ControlEval. Can you run ControlAgent with a small LLM at the level of 1.5B and report the performance on ControlEval? (You need to figure out from the Github repo how the evaluation is automatically done).
2. Clearly 1.5B model will not give a satisfying performance. Can you do something to improve the performance of ControlAgent with small LMs?
3. Does your method generalize to other engineering design problems? Say circuit design? (https://arxiv.org/pdf/2405.14918) Github:https://github.com/laiyao1/AnalogCoder

## My Plan
1️⃣ For the Task 1
1. Update the model invocation method by switching from the OpenAI API to local inference using Qwen-1.5B with Hugging Face Transformers.
2. Implement evaluation scripts to aggregate output results for SAR performance testing. 

2️⃣ For the Task 2
3. Identify areas where the 1.5B LLM underperforms across control and task-specific agents, then analyze the failure causes. 
4. Focus on fine-tuning or prompt weaknesses to improve performance.

2️⃣ For the Task 3
5. Essentially, it’s about designing different workflows for the LLM to process data and invoke tools, based on the specific tasks.

## Key Report



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
