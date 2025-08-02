# 🚀 Mistral 7B Domain Generator - Setup Guide

## 📋 Pre-requisites

### 1. Environment Setup
- **RunPod instance** with GPU (T4 or better)
- **Python 3.8+** 
- **CUDA available** for GPU acceleration

### 2. API Keys Required (RunPod Secrets)
Set up API keys as RunPod secrets for secure access:

**Method 1: RunPod Secrets (Recommended)**
1. Go to RunPod Console → Secrets
2. Create two secrets:
   - Secret Name: `HF_TOKEN` → Value: your HuggingFace token
   - Secret Name: `OPENAI_API_KEY` → Value: your OpenAI API key
3. In environment variables section of your pod, add:
   - `{{ RUNPOD_SECRET_HF_TOKEN }}`
   - `{{ RUNPOD_SECRET_OPENAI_API_KEY }}`

**Method 2: .env file (Alternative)**
Create a `.env` file in the same directory as the notebook:
```bash
HF_TOKEN=hf_your_huggingface_token_here
OPENAI_API_KEY=sk-your_openai_api_key_here
```

**How to get API keys:**
- **HuggingFace Token**: Go to https://huggingface.co/settings/tokens
- **OpenAI API Key**: Go to https://platform.openai.com/api-keys

## 🏃‍♂️ Quick Start

### Step 1: Upload Files
Upload these files to your RunPod instance:
- `mistral_domain_generation.ipynb`
- `.env` (with your API keys)

### Step 2: Install Dependencies
The notebook will automatically install required packages, but you can pre-install:

```bash
pip install transformers datasets peft torch tqdm pandas numpy matplotlib \
    python-Levenshtein gradio openai wandb python-dotenv huggingface_hub \
    seaborn plotly accelerate bitsandbytes scikit-learn
```

### Step 3: Run the Notebook
Execute cells sequentially. Expected runtime:
- **Cell 1-4**: Setup (2-3 minutes)
- **Cell 5**: Dataset creation (5-10 minutes)
- **Cell 6-7**: Model loading (3-5 minutes)  
- **Cell 8-9**: Fine-tuning (20-30 minutes)
- **Cell 10-11**: Evaluation (10-15 minutes)
- **Cell 12**: Gradio demo (1-2 minutes)

**Total runtime: ~45-60 minutes**

## 📊 Expected Outputs

### Dataset Creation
```
🎯 Creating 200 synthetic training examples using OpenAI...
✅ Dataset created with 200 samples
📊 Categories: 12
```

### Model Training
```
🚀 Starting 5-epoch training...
📊 Model parameters:
   Trainable parameters: 8,388,608
   Total parameters: 7,241,732,096
   Trainable %: 0.12%
```

### Evaluation Results
```
📊 Evaluation Results:
   baseline_validity: 0.850
   finetuned_validity: 0.920
   📈 validity_improvement: +0.070
```

### Gradio Demo
```
🌐 Ready to launch demo!
   Use: demo.launch(share=True) for public link
   Perfect for interview presentations!
```

## 🛠️ Troubleshooting

### Common Issues

**1. GPU Memory Error**
```python
# Solution: Reduce batch size in training arguments
per_device_train_batch_size=2  # Instead of 4
```

**2. API Rate Limits**
```python
# Solution: Reduce sample sizes
sample_size=5  # Instead of 10 for GPT-4 evaluation
```

**3. Model Download Issues**
```bash
# Solution: Pre-download model
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
```

**4. W&B Login Required**
```bash
# Optional: Login to Weights & Biases
wandb login
```

## 📁 Generated Files

After successful execution, you'll have:
- `domain_dataset.csv` - Training dataset
- `mistral_domain_technical_report.md` - Technical report
- `./mistral_domain_checkpoints/` - Model checkpoints
- `./mistral_domain_final/` - Final trained model
- `project_metadata.json` - Project metadata

## 🎯 Interview Demo

### Launch Gradio Interface
```python
# In the last cell, run:
demo.launch(share=True)
```

This creates a public URL you can share for remote demonstrations.

### Key Demo Points
1. **Safety filtering** - Try inappropriate content
2. **Model comparison** - Show baseline vs fine-tuned
3. **Business relevance** - Test with different business types
4. **Professional quality** - Demonstrate domain validity

## 💡 Interview Discussion Points

Be ready to discuss:
- **Why Mistral 7B**: Open source, performance, efficiency
- **Why LoRA**: Parameter efficiency, memory optimization
- **Why 5 epochs**: Balance of training time and performance
- **Evaluation methodology**: Multi-metric + LLM judge
- **Safety implementation**: Proactive content filtering
- **Production readiness**: Scalability and deployment

## 🚨 Important Notes

- **Cost**: OpenAI API calls will incur charges (~$5-10 for full run)
- **Time**: Full execution takes 45-60 minutes
- **GPU**: Requires 16GB+ VRAM (T4/V100/A10/A100)
- **Storage**: Needs ~10GB for model downloads and checkpoints

## ✅ Success Checklist

Before your interview, ensure:
- [ ] All cells execute without errors
- [ ] Technical report is generated
- [ ] Gradio demo launches successfully
- [ ] Model checkpoints are saved
- [ ] Evaluation metrics look reasonable

## 🆘 Support

If you encounter issues:
1. Check GPU memory usage: `nvidia-smi`
2. Verify API keys in `.env` file
3. Ensure stable internet for model downloads
4. Monitor disk space for checkpoints

Good luck with your interview! 🍀