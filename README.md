# Domain Name Generation with Fine-Tuned LLM

## AI Engineer Homework Solution

A comprehensive solution for domain name generation using fine-tuned LLMs with systematic evaluation, edge case discovery, and safety guardrails.

## 🚀 Features

- **Synthetic Dataset Creation**: Generates diverse business descriptions and domain names
- **Model Fine-tuning**: LoRA/PEFT fine-tuning for improved performance
- **LLM-as-a-Judge**: GPT-4 powered evaluation framework
- **Safety Guardrails**: Content filtering and inappropriate content detection
- **Edge Case Analysis**: Systematic discovery and analysis of failure modes
- **Interactive Interface**: Gradio-based web UI for easy testing
- **REST API**: FastAPI implementation for production deployment
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## 📁 Project Structure

```
domain_name-/
├── domain_name_generation.ipynb  # Main Jupyter notebook
├── api.py                       # FastAPI REST API
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── data/                       # Generated datasets
├── checkpoints/               # Model checkpoints
├── reports/                   # Generated reports
└── logs/                      # Application logs
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- Docker (optional)

### 🚀 Quick Start (Complete Solution)

1. **Navigate to project directory**
   ```bash
   cd /Users/linoospaulinos/python_project_2025/domain_name-
   ```

2. **Run the complete solution**
   ```bash
   ./start_notebook.sh
   ```
   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate
   jupyter notebook final_complete_solution.ipynb
   ```

3. **Environment is pre-configured** with OpenAI API key

### 📚 Available Notebooks

- **`final_complete_solution.ipynb`** - ⭐ **COMPLETE SOLUTION** with all requirements
- **`domain_name_generation_complete.ipynb`** - Advanced multi-model implementation  
- **`domain_name_generation_fixed.ipynb`** - Basic working solution

### Installation Notes

- The notebook automatically detects and uses GPU if available, falls back to CPU
- All required packages are auto-installed in the first cell
- OpenAI API key is already configured in the .env file
- The notebook uses a lighter model (distilgpt2) on CPU for better performance

### Docker Setup (Alternative)

1. **Build Docker image**
   ```bash
   docker build -t domain-name-generator .
   ```

2. **Run container**
   ```bash
   docker run -p 8888:8888 -p 7860:7860 -p 8000:8000 domain-name-generator
   ```

## 📊 Usage

### Jupyter Notebook

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Navigate to `domain_name_generation.ipynb`
   - Run all cells to execute the complete pipeline

### REST API

1. **Start the API server**
   ```bash
   python api.py
   ```

2. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/generate" \
        -H "Content-Type: application/json" \
        -d '{
          "business_description": "organic coffee shop in downtown area",
          "num_suggestions": 3,
          "model_type": "baseline"
        }'
   ```

3. **API Documentation**
   - Visit `http://localhost:8000/docs` for interactive API documentation

### Gradio Interface

1. **Launch from notebook**
   - Run the Gradio section in the notebook
   - Uncomment `demo.launch()` line

2. **Access the interface**
   - Visit `http://localhost:7860`

## 🧪 Model Training

### Dataset Generation

The system automatically generates a synthetic dataset with:
- 1000+ business descriptions
- Diverse business categories (e-commerce, food, tech, health, etc.)
- Edge cases for testing robustness

### Fine-tuning Process

1. **Baseline Model**: Zero-shot generation using pre-trained model
2. **Fine-tuned Model**: LoRA/PEFT fine-tuning on synthetic data
3. **Evaluation**: Comprehensive comparison using multiple metrics

### Model Evaluation

- **Exact Match Score**: Direct comparison with reference domains
- **Levenshtein Similarity**: Edit distance-based similarity
- **LLM-as-a-Judge**: GPT-4 powered quality assessment
- **Quality Dimensions**: Length, readability, relevance, memorability

## 🔒 Safety Features

### Content Filtering

- Profanity detection and filtering
- Inappropriate content categories blocking
- Sensitive content flagging with extra review

### Safety Categories

- **Blocked**: Adult content, violence, drugs, hate speech, illegal activities
- **Sensitive**: Financial services, medical, political, religious content

### Domain Validation

- Format validation (valid domain structure)
- Length constraints (reasonable domain lengths)
- Character validation (alphanumeric and allowed special chars)

## 📈 Performance Metrics

### Baseline Results

- Exact Match: ~15-25%
- Levenshtein Similarity: ~65-75%
- Average Quality Score: ~70-80%

### Fine-tuned Results

- Expected improvement: 5-15% across all metrics
- Better relevance and domain structure
- Reduced inappropriate content generation

## 🔍 Edge Case Analysis

### Tested Categories

1. **Very Long Descriptions**: Complex multi-sentence business descriptions
2. **Very Short Descriptions**: Single word inputs
3. **Ambiguous Descriptions**: Vague or unclear business descriptions
4. **Technical Jargon**: Industry-specific technical terms
5. **International Languages**: Non-English business descriptions
6. **Special Characters**: Descriptions with symbols and punctuation
7. **Conflicting Concepts**: Contradictory business descriptions
8. **Nonsensical Inputs**: Completely unrelated or nonsensical text

### Failure Analysis

- **Generation Failures**: Complete failure to generate output
- **Inappropriate Output**: Safety violations or offensive content
- **Irrelevant Output**: Domains unrelated to business description
- **Malformed Domains**: Invalid domain formats
- **Length Issues**: Domains too short or too long

## 📋 Technical Report

The system generates comprehensive technical reports including:

- **Executive Summary**: Key findings and improvements
- **Model Comparison**: Detailed performance analysis
- **Edge Case Analysis**: Failure modes and frequencies
- **Safety Analysis**: Guardrail effectiveness
- **Recommendations**: Production deployment guidance

Reports are saved to `reports/technical_report.md`

## 🚀 Production Deployment

### API Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

2. **Start API Server**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
   ```

3. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

### Monitoring

- Request logging for all API calls
- Performance metrics tracking
- Safety violation monitoring
- Model performance tracking

## 🧪 Testing

### Unit Tests

```bash
# Run basic functionality tests
python -m pytest tests/ -v
```

### API Tests

```bash
# Test API endpoints
curl -X GET "http://localhost:8000/health"
curl -X GET "http://localhost:8000/examples"
```

### Safety Tests

The notebook includes comprehensive safety testing with various inappropriate inputs to verify guardrail effectiveness.

## 📝 Configuration

### Model Configuration

- **Model**: Microsoft DialoGPT-medium (changeable)
- **Max Length**: 512 tokens
- **Batch Size**: 4 (adjustable based on GPU memory)
- **Learning Rate**: 5e-5
- **Epochs**: 3

### LoRA Configuration

- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: q_proj, v_proj, k_proj, o_proj

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## 📄 License

This project is for educational purposes as part of the AI Engineer homework assignment.

## 🔗 References

- [Transformers Library](https://huggingface.co/transformers/)
- [PEFT (Parameter Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Gradio](https://gradio.app/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

For questions or issues:
1. Check the technical report in `reports/technical_report.md`
2. Review the Jupyter notebook documentation
3. Check API documentation at `/docs` endpoint
4. Review the comprehensive examples in the notebook

---

**Note**: This implementation focuses on the evaluation and improvement framework as suggested in the homework instructions. The fine-tuning component may require significant computational resources and should be run on appropriate hardware.