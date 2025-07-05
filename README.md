# Thinking Beyond Tokens: From Brain-Inspired Intelligence to Cognitive Foundations for Artificial General Intelligence and its Societal Impact

[![arXiv](https://img.shields.io/badge/arXiv-2507.00951-b31b1b.svg)](https://arxiv.org/abs/2507.00951)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

> **Abstract**: Can machines truly think, reason and act in domains like humans? This enduring question continues to shape the pursuit of Artificial General Intelligence (AGI). Despite the growing capabilities of models such as GPT-4.5, DeepSeek, Claude 3.5 Sonnet, Phi-4, and Grok 3, which exhibit multimodal fluency and partial reasoning, these systems remain fundamentally limited by their reliance on token-level prediction and lack of grounded agency.

## 📖 About This Research

This paper offers a **cross-disciplinary synthesis** of AGI development, spanning artificial intelligence, cognitive neuroscience, psychology, generative models, and agent-based systems. We analyze the architectural and cognitive foundations of general intelligence, highlighting the role of modular reasoning, persistent memory, and multi-agent coordination.

### 🎯 Key Contributions

- **Unified Framework**: Synthesizes insights from neuroscience, cognition, and AI to identify foundational principles for AGI system design
- **Critical Analysis**: Examines limitations of current token-level models and post hoc alignment strategies
- **Emergent Methods Survey**: Covers modular cognition, world modeling, neuro-symbolic reasoning, and biologically inspired architectures
- **Multidimensional Roadmap**: Presents a comprehensive path for AGI development incorporating logical reasoning, lifelong learning, embodiment, and ethical oversight
- **Cognitive Function Mapping**: Maps core human cognitive functions to computational analogues

## 🧠 Core Concepts

### Why Token-Level Prediction Alone is Insufficient for AGI

Current models like GPT-4, DeepSeek, and Grok capture surface linguistic patterns but fail to support complex mental representations grounded in the physical world. Lacking embodiment, causality, and self-reflection, they struggle with abstraction and goal-directed behavior—core requirements for AGI.

### Beyond Scaling: The Need for Architectural Innovation

While scaling improves fluency and performance on many tasks, it cannot resolve core limitations of current LLMs. These models still lack:
- Grounded understanding
- Causal reasoning  
- Persistent memory
- Goal-directed behavior

## 🚀 Research Highlights

### 🎭 Reasoning Systems

| System | Date | Key Innovation | Links | Status |
|:-------|------|----------------|-------|--------|
| **Generative Agents** | Apr 2023 | Simulate human behavior with AI agents | [[Paper](https://arxiv.org/abs/2304.03442)] [[Demo](https://reverie.herokuapp.com/arXiv_Demo/)] [[Code](https://github.com/joonspk-research/generative_agents)] | ✅ Available |
| **AutoGPT** | Apr 2023 | Objective-driven execution with agents | [[GitHub](https://github.com/Significant-Gravitas/Auto-GPT)] [[Try Online](https://godmode.space/)] | ✅ Available |
| **BabyAGI** | Apr 2023 | Task expander loop architecture | [[GitHub](https://github.com/yoheinakajima/babyagi)] [[Article](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)] | ✅ Available |
| **MetaGPT** | 2023 | Multi-agent framework for software development | [[GitHub](https://github.com/geekan/MetaGPT)] [[Paper](https://arxiv.org/abs/2308.00352)] | ✅ Available |
| **ReAct** | 2022 | Synergizing reasoning and acting | [[Paper](https://arxiv.org/abs/2210.03629)] [[GitHub](https://github.com/ysymyth/ReAct)] | ✅ Available |
| **HuggingGPT/JARVIS** | Mar 2023 | Model calls specialized models for input | [[Paper](https://arxiv.org/abs/2303.17580)] [[GitHub](https://github.com/microsoft/JARVIS)] | ✅ Available |
| **Reflexion** | Mar 2023 | Autonomous agent with dynamic memory | [[Paper](https://arxiv.org/abs/2303.11366)] [[GitHub](https://github.com/noahshinn024/reflexion)] | ✅ Available |

### 🤖 Foundation Models & LLMs

| Model | Organization | Capabilities | Links | Status |
|:------|-------------|-------------|-------|--------|
| **GPT-4** | OpenAI | Multimodal language understanding | [[API](https://openai.com/gpt-4)] [[Paper](https://arxiv.org/abs/2303.08774)] | ✅ Available |
| **DeepSeek-V3** | DeepSeek | 236B MoE model with 128K context | [[Model](https://huggingface.co/deepseek-ai)] [[Paper](https://arxiv.org/abs/2412.19437)] | ✅ Available |
| **Claude 3.5 Sonnet** | Anthropic | Advanced reasoning and safety | [[API](https://www.anthropic.com/claude)] [[Model](https://huggingface.co/anthropic)] | ✅ Available |
| **Grok 3** | xAI | Real-time information processing | [[Platform](https://grok.x.ai/)] [[Paper](https://arxiv.org/abs/2502.16428)] | ✅ Available |
| **Phi-4** | Microsoft | Efficient small language model | [[Model](https://huggingface.co/microsoft/phi-4)] [[Paper](https://arxiv.org/abs/2412.08905)] | ✅ Available |
| **Gemini** | Google | Multimodal AI system | [[API](https://ai.google.dev/)] [[Paper](https://arxiv.org/abs/2312.11805)] | ✅ Available |
| **LLaMA** | Meta | Open foundation language models | [[Model](https://huggingface.co/meta-llama)] [[Paper](https://arxiv.org/abs/2302.13971)] | ✅ Available |

### 🖼️ Vision-Language Models (VLMs)

| Model | Organization | Key Features | Links | Status |
|:------|-------------|-------------|-------|--------|
| **GPT-4V** | OpenAI | Vision-language understanding | [[API](https://openai.com/gpt-4)] [[Docs](https://platform.openai.com/docs/guides/vision)] | ✅ Available |
| **Gemini 2.5 Pro** | Google | Advanced multimodal reasoning | [[API](https://ai.google.dev/gemini-api)] [[Docs](https://ai.google.dev/docs)] | ✅ Available |
| **LLaVA** | Various | Open-source vision-language model | [[GitHub](https://github.com/haotian-liu/LLaVA)] [[Model](https://huggingface.co/llava-hf)] | ✅ Available |
| **Qwen2.5-VL** | Alibaba | Multilingual vision-language model | [[Model](https://huggingface.co/Qwen/Qwen2.5-VL-32B)] [[GitHub](https://github.com/QwenLM/Qwen2-VL)] | ✅ Available |
| **InternVL** | OpenGVLab | Versatile vision-language model | [[GitHub](https://github.com/OpenGVLab/InternVL)] [[Model](https://huggingface.co/OpenGVLab/InternVL2-26B)] | ✅ Available |
| **CLIP** | OpenAI | Contrastive language-image pre-training | [[GitHub](https://github.com/openai/CLIP)] [[Model](https://huggingface.co/openai/clip-vit-base-patch32)] | ✅ Available |
| **Flamingo** | DeepMind | Few-shot learning for vision-language | [[Paper](https://arxiv.org/abs/2204.14198)] [[Unofficial Code](https://github.com/lucidrains/flamingo-pytorch)] | 📄 Paper Only |

### 🧪 Research Frameworks & Platforms

| Framework | Type | Description | Links | Status |
|:----------|------|-------------|-------|--------|
| **AutoGPT** | Agent Framework | Autonomous task execution | [[GitHub](https://github.com/Significant-Gravitas/Auto-GPT)] [[Try Online](https://agentgpt.reworkd.ai/)] | ✅ Available |
| **MetaGPT** | Multi-Agent | Software development agents | [[GitHub](https://github.com/geekan/MetaGPT)] [[Demo](https://github.com/geekan/MetaGPT#-quickstart)] | ✅ Available |
| **SuperAGI** | Agent Platform | Build and run autonomous agents | [[GitHub](https://github.com/TransformerOptimus/SuperAGI)] [[Docs](https://superagi.com/docs)] | ✅ Available |
| **AgentGPT** | Web Platform | Browser-based autonomous agents | [[GitHub](https://github.com/reworkd/AgentGPT)] [[Try Online](https://agentgpt.reworkd.ai/)] | ✅ Available |
| **LangChain** | Framework | Building LLM applications | [[GitHub](https://github.com/langchain-ai/langchain)] [[Docs](https://docs.langchain.com/)] | ✅ Available |
| **OpenAGI** | Framework | Domain expert integration | [[GitHub](https://github.com/agiresearch/OpenAGI)] [[Paper](https://arxiv.org/abs/2304.04370)] | ✅ Available |

### 🤖 Autonomous AI Agents

| Agent | Organization | Specialization | Links | Status |
|:------|-------------|---------------|-------|--------|
| **Voyager** | NVIDIA/Caltech | Minecraft exploration | [[GitHub](https://github.com/MineDojo/Voyager)] [[Paper](https://arxiv.org/abs/2305.16291)] | ✅ Available |
| **GPT-Engineer** | AntonOsika | Full-stack development | [[GitHub](https://github.com/AntonOsika/gpt-engineer)] [[Docs](https://gpt-engineer.readthedocs.io/)] | ✅ Available |
| **GPT-Researcher** | AssafElovic | Comprehensive research | [[GitHub](https://github.com/assafelovic/gpt-researcher)] [[Demo](https://gptr.dev/)] | ✅ Available |
| **AutoGen** | Microsoft | Multi-agent conversations | [[GitHub](https://github.com/microsoft/autogen)] [[Docs](https://microsoft.github.io/autogen/)] | ✅ Available |
| **CrewAI** | CrewAI | Role-playing multi-agent teams | [[GitHub](https://github.com/joaomdmoura/crewAI)] [[Docs](https://docs.crewai.com/)] | ✅ Available |
| **AI Town** | a16z | AI agent simulation environment | [[GitHub](https://github.com/a16z-infra/ai-town)] [[Demo](https://www.convex.dev/ai-town)] | ✅ Available |

### 🧬 Brain-Inspired Architectures

| Architecture | Type | Key Innovation | Links | Status |
|:-------------|------|---------------|-------|--------|
| **Spiking Neural Networks** | Neuromorphic | Emulate neural spike dynamics | [[BindsNET](https://github.com/BindsNET/bindsnet)] [[NEST](https://www.nest-simulator.org/)] [[Brian2](https://github.com/brian-team/brian2)] | ✅ Available |
| **Physics-Informed Neural Networks** | Hybrid | Incorporate physical laws into NNs | [[DeepXDE](https://github.com/lululxvi/deepxde)] [[PINN Papers](https://github.com/lu-group/pinn-bibliography)] | ✅ Available |
| **Kolmogorov-Arnold Networks** | Novel Architecture | Learnable spline-based activations | [[PyKAN](https://github.com/KindXiaoming/pykan)] [[Paper](https://arxiv.org/abs/2404.19756)] | ✅ Available |
| **Neural ODEs** | Continuous | Continuous-time neural networks | [[torchdiffeq](https://github.com/rtqichen/torchdiffeq)] [[Paper](https://arxiv.org/abs/1806.07366)] | ✅ Available |
| **Liquid Neural Networks** | Adaptive | Dynamic, adaptable neural circuits | [[ncps](https://github.com/mlech26l/ncps)] [[Paper](https://arxiv.org/abs/2006.04439)] | ✅ Available |
| **Neural Turing Machines** | Memory-Augmented | External memory mechanisms | [[PyTorch NTM](https://github.com/loudinthecloud/pytorch-ntm)] [[Paper](https://arxiv.org/abs/1410.5401)] | ✅ Available |

### 🎯 Specialized AI Models

| Model Type | Examples | Purpose | Links | Status |
|:-----------|----------|---------|-------|--------|
| **Large Concept Models** | SONAR, Concept-LLMs | Concept-level reasoning | [[SONAR](https://github.com/facebookresearch/LASER)] [[Paper](https://arxiv.org/abs/2412.08821)] | ✅ Available |
| **Large Reasoning Models** | OpenAI o1, DeepSeek-R1 | Extended inference-time reasoning | [[OpenAI o1](https://openai.com/o1/)] [[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)] | ✅ Available |
| **Mixture of Experts** | Switch Transformer, GLaM | Sparse expert routing | [[Switch Transformer](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer)] [[Paper](https://arxiv.org/abs/2101.03961)] | ✅ Available |
| **Retrieval-Augmented** | RAG, RETRO, Atlas | External knowledge integration | [[LangChain RAG](https://github.com/langchain-ai/langchain)] [[RETRO](https://github.com/lucidrains/RETRO-pytorch)] | ✅ Available |
| **World Models** | DreamerV3, MuZero | Environment modeling | [[DreamerV3](https://github.com/danijar/dreamerv3)] [[MuZero](https://github.com/werner-duvaud/muzero-general)] | ✅ Available |

### 🔬 Benchmark Datasets & Evaluation

| Benchmark | Focus | Description | Links | Status |
|:----------|-------|-------------|-------|--------|
| **BIG-Bench** | Language Reasoning | 200+ diverse language tasks | [[GitHub](https://github.com/google/BIG-bench)] [[Paper](https://arxiv.org/abs/2206.04615)] | ✅ Available |
| **ARC** | Abstract Reasoning | Visual pattern recognition | [[GitHub](https://github.com/fchollet/ARC)] [[Dataset](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)] | ✅ Available |
| **MineDojo** | Embodied AI | Minecraft-based embodied learning | [[GitHub](https://github.com/MineDojo/MineDojo)] [[Website](https://minedojo.org/)] | ✅ Available |
| **AgentBench** | LLM Agents | Multi-domain agent evaluation | [[GitHub](https://github.com/THUDM/AgentBench)] [[Paper](https://arxiv.org/abs/2308.03688)] | ✅ Available |
| **AGI-Bench** | General Intelligence | Multimodal AGI evaluation | [[GitHub](https://github.com/Dawn-LX/AGI-Bench)] [[Paper](https://arxiv.org/abs/2305.07153)] | ✅ Available |
| **HELM** | Language Models | Holistic evaluation framework | [[GitHub](https://github.com/stanford-crfm/helm)] [[Website](https://crfm.stanford.edu/helm/)] | ✅ Available |
| **MMMU** | Multimodal Understanding | College-level multimodal tasks | [[GitHub](https://github.com/MMMU-Benchmark/MMMU)] [[Website](https://mmmu-benchmark.github.io/)] | ✅ Available |

### 🛠️ Development Tools & Libraries

| Tool | Category | Purpose | Links | Status |
|:-----|----------|---------|-------|--------|
| **Transformers** | Model Library | Hugging Face model hub | [[GitHub](https://github.com/huggingface/transformers)] [[Docs](https://huggingface.co/docs/transformers)] | ✅ Available |
| **LangChain** | Framework | LLM application development | [[GitHub](https://github.com/langchain-ai/langchain)] [[Docs](https://docs.langchain.com/)] | ✅ Available |
| **LlamaIndex** | RAG Framework | Data framework for LLMs | [[GitHub](https://github.com/run-llama/llama_index)] [[Docs](https://docs.llamaindex.ai/)] | ✅ Available |
| **OpenAI Gym** | RL Environment | Reinforcement learning toolkit | [[GitHub](https://github.com/openai/gym)] [[Website](https://gym.openai.com/)] | ✅ Available |
| **PettingZoo** | Multi-Agent RL | Multi-agent RL environments | [[GitHub](https://github.com/Farama-Foundation/PettingZoo)] [[Docs](https://pettingzoo.farama.org/)] | ✅ Available |
| **Ray** | Distributed Computing | Scalable ML and AI workloads | [[GitHub](https://github.com/ray-project/ray)] [[Docs](https://docs.ray.io/)] | ✅ Available |
| **Weights & Biases** | MLOps | Experiment tracking and MLOps | [[GitHub](https://github.com/wandb/wandb)] [[Platform](https://wandb.ai/)] | ✅ Available |

### 🌐 Online Demos & Platforms

| Platform | Type | Description | Links | Access |
|:---------|------|-------------|-------|--------|
| **ChatGPT** | Conversational AI | OpenAI's flagship chatbot | [[Platform](https://chat.openai.com/)] | 🔓 Free/Paid |
| **Claude** | Conversational AI | Anthropic's AI assistant | [[Platform](https://claude.ai/)] | 🔓 Free/Paid |
| **Bard/Gemini** | Conversational AI | Google's AI assistant | [[Platform](https://bard.google.com/)] | 🔓 Free |
| **AgentGPT** | Autonomous Agents | Browser-based agent creation | [[Platform](https://agentgpt.reworkd.ai/)] | 🔓 Free |
| **Godmode** | AutoGPT Interface | User-friendly AutoGPT interface | [[Platform](https://godmode.space/)] | 🔓 Free |
| **Cognosys** | AI Agents | AI agent automation platform | [[Platform](https://www.cognosys.ai/)] | 🔓 Free/Paid |
| **AI Town Demo** | Agent Simulation | Generative agents in virtual town | [[Demo](https://www.convex.dev/ai-town)] | 🔓 Free |

### 📚 Educational Resources & Courses

| Resource | Type | Focus | Links | Access |
|:---------|------|-------|-------|--------|
| **CS231n** | Course | Convolutional Neural Networks | [[Stanford](http://cs231n.stanford.edu/)] [[YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)] | 🔓 Free |
| **CS224n** | Course | Natural Language Processing | [[Stanford](http://web.stanford.edu/class/cs224n/)] [[YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)] | 🔓 Free |
| **Deep Learning Book** | Textbook | Comprehensive deep learning | [[Online](https://www.deeplearningbook.org/)] [[PDF](https://github.com/janishar/mit-deep-learning-book-pdf)] | 🔓 Free |
| **AGI Safety Fundamentals** | Course | AI safety and alignment | [[Curriculum](https://www.aisafetyfundamentals.com/)] [[Materials](https://www.aisafetyfundamentals.com/agi-safety-fundamentals)] | 🔓 Free |
| **Neurosymbolic AI** | Course | Hybrid AI approaches | [[MIT](https://people.csail.mit.edu/jda/teaching/6.S099/)] [[Materials](https://github.com/neurosymbolic-learning/Neurosymbolic_Tutorial)] | 🔓 Free |

### 🔄 Generalization Frameworks & Theory

| Framework | Type | Key Insight | Links | Status |
|:----------|------|-------------|-------|--------|
| **Information Bottleneck** | Theory | Compression enables generalization | [[Paper](https://arxiv.org/abs/physics/0004057)] [[Implementation](https://github.com/artemyk/ibsgd)] | ✅ Available |
| **Neural Tangent Kernel** | Theory | Infinite-width network behavior | [[Paper](https://arxiv.org/abs/1806.07572)] [[JAX Implementation](https://github.com/google/neural-tangents)] | ✅ Available |
| **PAC-Bayes** | Theory | Generalization bounds | [[Tutorial](https://github.com/john-bradshaw/PAC-Bayes-tutorial)] [[PyTorch](https://github.com/paulviallard/PacBayesianNeuralNetwork)] | ✅ Available |
| **Causal Representation** | Framework | Causal structure learning | [[CausalML](https://github.com/uber/causalml)] [[DoWhy](https://github.com/microsoft/dowhy)] | ✅ Available |
| **Meta-Learning** | Framework | Learning to learn | [[MAML](https://github.com/cbfinn/maml)] [[learn2learn](https://github.com/learnables/learn2learn)] | ✅ Available |

## 🎯 Key Research Areas

### 1. Cognitive Architecture Design
- Modular reasoning systems
- Persistent memory mechanisms
- Multi-agent coordination
- World model integration

### 2. Learning Paradigms
- Meta-learning and continual learning
- Few-shot and zero-shot generalization
- Causal representation learning
- Uncertainty quantification

### 3. Alignment and Safety
- Human-in-the-loop training
- Value learning and preference optimization
- Ethical framework integration
- Transparency and interpretability

### 4. Societal Integration
- Democratic AI development
- Cultural sensitivity and inclusion
- Economic impact assessment
- Governance framework design

## 🔬 Experimental Insights

### Large Concept Models (LCMs)
Moving beyond token-level processing to concept-level reasoning, operating over explicit semantic representations that are language and modality-agnostic.

### Large Reasoning Models (LRMs)
Systems focused on explicit, multi-step cognitive processes rather than single-shot response generation, employing extended inference time computation.

### Agentic AI Systems
Autonomous systems with planning, memory, tool-use, and decision-making capabilities that mirror core aspects of human cognition.

## 📈 Future Directions

### Missing Pieces in Current AGI Development

1. **Uncertainty Management**: Handling both epistemic and aleatory uncertainty
2. **Compression-Based Reasoning**: Moving beyond memorization to true abstraction
3. **Emotional Intelligence**: Understanding and navigating social dynamics
4. **Ethical Framework**: Embedding moral reasoning from design inception
5. **Environmental Sustainability**: Energy-efficient architectures and operations

### Emerging Paradigms

- **Neural Society of Agents**: Distributed intelligence through agent collaboration
- **Mixture of Experts**: Specialized sub-networks for different cognitive functions
- **Self-Evolving Systems**: Agents that autonomously improve their reasoning processes

## 🤝 Contributing

We welcome contributions from researchers across disciplines! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Submitting improvements to cognitive architectures
- Adding new benchmark evaluations
- Proposing ethical framework enhancements
- Sharing experimental results

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{qureshi2025thinking,
  title={Thinking Beyond Tokens: From Brain-Inspired Intelligence to Cognitive Foundations for Artificial General Intelligence and its Societal Impact},
  author={Qureshi, Rizwan and Sapkota, Ranjan and Shah, Abbas and Muneer, Amgad and Zafar, Anas and Vayani, Ashmal and others},
  journal={arXiv preprint arXiv:2507.00951},
  year={2025}
}
```

## 👥 Complete Author List
- **Rizwan Qureshi**¹* - Center for Research in Computer Vision, University of Central Florida
- **Ranjan Sapkota**²* - Department of Biological and Environmental Engineering, Cornell University
- **Abbas Shah**³* - Department of Electronics Engineering, Mehran University of Engineering & Technology
- **Amgad Muneer**⁴* - Department of Imaging Physics, The University of Texas MD Anderson Cancer Center
- **Anas Zafar**⁴ - Department of Imaging Physics, The University of Texas MD Anderson Cancer Center
- **Ashmal Vayani**¹ - Center for Research in Computer Vision, University of Central Florida
- **Maged Shoman**⁵ - Intelligent Transportation Systems, University of Tennessee
- **Abdelrahman B. M. Eldaly**⁶ - Department of Electrical Engineering, City University of Hong Kong
- **Kai Zhang**⁴ - Department of Imaging Physics, The University of Texas MD Anderson Cancer Center
- **Ferhat Sadak**⁷ - Department of Mechanical Engineering, Bartin University
- **Shaina Raza**⁸† - Vector Institute, Toronto (Corresponding Author)
- **Xinqi Fan**⁹ - Manchester Metropolitan University
- **Ravid Shwartz-Ziv**¹⁰ - Center for Data Science, New York University
- **Hong Yan**⁶ - Department of Electrical Engineering, City University of Hong Kong
- **Vinjia Jain**¹¹ - Meta Research (Work done outside Meta)
- **Aman Chadha**¹² - Amazon Research (Work done outside Amazon)
- **Manoj Karkee**² - Department of Biological and Environmental Engineering, Cornell University
- **Jia Wu**⁴ - Department of Imaging Physics, The University of Texas MD Anderson Cancer Center
- **Philip Torr**¹³ - Department of Engineering Science, University of Oxford
- **Seyedali Mirjalili**¹⁴,¹⁵ - Centre for Artificial Intelligence Research and Optimization, Torrens University Australia & University Research and Innovation Center, Obuda University

*Equal Contribution | †Corresponding Author: shaina.raza@torontomu.ca

## 🏛️ Complete Institutional Affiliations

### 🇺🇸 United States
- **¹ University of Central Florida** - Center for Research in Computer Vision, Orlando, FL
- **² Cornell University** - Department of Biological and Environmental Engineering, Ithaca, NY
- **⁴ The University of Texas MD Anderson Cancer Center** - Department of Imaging Physics, Houston, TX
- **⁵ University of Tennessee** - Intelligent Transportation Systems, Oak Ridge, TN
- **¹⁰ New York University** - Center for Data Science, New York, NY
- **¹¹ Meta Research** - (Work done outside Meta)
- **¹² Amazon Research** - (Work done outside Amazon)

### 🇨🇦 Canada
- **⁸ Vector Institute** - Toronto, Canada

### 🇬🇧 United Kingdom
- **⁹ Manchester Metropolitan University** - Manchester, UK
- **¹³ University of Oxford** - Department of Engineering Science, UK

### 🇭🇰 Hong Kong (SAR China)
- **⁶ City University of Hong Kong** - Department of Electrical Engineering

### 🇵🇰 Pakistan
- **³ Mehran University of Engineering & Technology** - Department of Electronics Engineering, Jamshoro, Sindh

### 🇹🇷 Turkey
- **⁷ Bartin University** - Department of Mechanical Engineering, Bartin

### 🇦🇺 Australia
- **¹⁴ Torrens University Australia** - Centre for Artificial Intelligence Research and Optimization, Fortitude Valley, Brisbane, QLD

### 🇭🇺 Hungary
- **¹⁵ Obuda University** - University Research and Innovation Center, Budapest

## 🌍 Global Collaboration Summary

This research represents a truly **international collaboration** spanning:
- **8 Countries**: United States, Canada, United Kingdom, Hong Kong, Pakistan, Turkey, Australia, Hungary
- **15 Institutions**: Leading universities and research centers worldwide
- **16 Authors**: Experts from diverse fields including AI, neuroscience, engineering, and cognitive science
- **Multiple Disciplines**: Computer Vision, AI Safety, Neuroscience, Engineering, Physics, and Philosophy

### Research Domains Represented
- 🧠 **Cognitive Neuroscience & Psychology**
- 🤖 **Artificial Intelligence & Machine Learning**
- 🔬 **Computer Vision & Multimodal AI**
- ⚡ **Engineering & Optimization**
- 🛡️ **AI Safety & Ethics**
- 🏥 **Medical Physics & Imaging**
- 🚗 **Intelligent Transportation Systems**
- 🔧 **Biological & Environmental Engineering**

## 📧 Contact

For questions, collaborations, or discussions:
- **Corresponding Author**: [shaina.raza@torontomu.ca](mailto:shaina.raza@torontomu.ca)
- **GitHub Issues**: For technical questions and bug reports
- **Discussions**: For research discussions and ideas

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the global AI research community for their foundational contributions to understanding intelligence, consciousness, and the path toward AGI. Special recognition to the institutions and funding bodies that supported this interdisciplinary research effort.

---

**"True intelligence arises not from scale alone but from the integration of memory and reasoning: an orchestration of modular, interactive, and self-improving components where compression enables adaptive behavior."**

*— From the paper*
