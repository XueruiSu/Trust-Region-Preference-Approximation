<div align="center">

# Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning

<a href="https://arxiv.org/abs/2504.04524"><img src="https://img.shields.io/badge/📝-Paper-orange"></a>
<a href="https://github.com/XueruiSu/Trust-Region-Preference-Approximation/blob/main/LICENSE"><img src="https://img.shields.io/github/license/XueruiSu/Trust-Region-Preference-Approximation"></a>
</div>


We propose the Trust Region Preference Approximation (TRPA) algorithm ⚙️, which integrates rule-based optimization with preference-based optimization for LLM reasoning tasks 🤖🧠. As a preference-based algorithm, TRPA naturally eliminates the reward hacking issue. TRPA constructs preference levels using predefined rules, forms corresponding preference pairs, and leverages a novel optimization algorithm for RL training with a theoretical monotonic improvement guarantee. Experimental results demonstrate that TRPA not only achieves competitive performance on reasoning tasks but also exhibits robust stability.

![TRPA](https://github.com/user-attachments/assets/7c975200-e618-4b1a-9e5e-e50ed1b9de7a)


## 🏆 Benchmark

<div align="center">
    
| Model                                                             | 2ppl | 3ppl | 4ppl | 5ppl | 6ppl | 7ppl | 8ppl |
|------------------------------------------------------------------------|------|------|------|------|------|------|------|
| o3-mini-high                | 0.99 | 0.98 | 0.97 | 0.95 | 0.94 | 0.89 | 0.83 |
| o1-2024-12-17               | 0.83 | 0.51 | 0.38 | 0.38 | 0.35 | 0.30 | 0.20 |
| GPT-4o                      | 0.68 | 0.57 | 0.49 | 0.32 | 0.23 | 0.21 | 0.11 |
| Deepseek-Math-7b            | 0.35 | 0.21 | 0.08 | 0.06 | 0.02 | 0.00 | 0.00 |
| Qwen2.5-7B-Instruct-1M      | 0.49 | 0.40 | 0.25 | 0.11 | 0.02 | 0.06 | 0.01 |
| Qwen2.5-7B-Logic-RL         | 0.99 | 0.99 | 0.94 | 0.92 | 0.91 | 0.80 | 0.67 |
| Qwen2.5-7B-TRPA (ours)      | 0.96 | 0.99 | 0.98 | 0.95 | 0.92 | 0.91 | 0.86 |

</div>

## 🛠️ Installation
```bash
conda create -n TRPA python=3.9
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib codetiming accelerate
pip install tensordict
pip install omegaconf hydra-core pylatexenc tabulate
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

## 📝 Data Preparation
```bash
python ./scripts/data_preprocess/kk_data_process.py \
    --template_type=qwen-instruct \ (Optional)
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

## 🦾 Training
```bash
conda activate TRPA
bash main_TRPA.sh  # 4×A100 80G
```

## 🤖 Evaluation

Our evaluation scripts automatically runs vLLM to generate 16 samples for each problem. To run our evaluation scripts, run:
```bash
./scripts/eval/eval_with_generation.sh --model [CHECKPOINT_PATH] --datasets [DATASET1] [DATASET2] --output-dir [OUTPUT_DIR]
```

## 📚 Citation
```bibtex
@article{su2025trustregionpreferenceapproximation,
  title={Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning}, 
  author={Xuerui Su and Shufang Xie and Guoqing Liu and Yingce Xia and Renqian Luo and Peiran Jin and Zhiming Ma and Yue Wang and Zun Wang and Yuting Liu},
  journal={arXiv:2504.04524},
  year={2025},
}
```

## 📖 Acknowledgements
- [Verl](https://arxiv.org/abs/2409.19256) 🔗
- [Logic RL](https://arxiv.org/abs/2502.14768) 🔗
- [Knights and Knaves (K&K) puzzles dataset](https://arxiv.org/abs/2410.23123) 🔗
- [DeepScaleR](https://github.com/agentica-project/deepscaler) 🔗





