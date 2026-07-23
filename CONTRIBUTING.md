# Contributing to FlickMatrix AI 🎬

Thank you for your interest in contributing to **FlickMatrix AI**! We welcome open-source contributions, bug reports, feature requests, and performance enhancements.

---

## 🚀 How to Contribute

1. **Fork the Repository** on GitHub.
2. **Clone your Fork** locally:
   ```bash
   git clone https://github.com/Akshay-gurav-31/flickmatrix-ai.git
   cd flickmatrix-ai
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/amazing-new-feature
   ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run Model Training & Verification**:
   ```bash
   python scripts/train.py --force-prep
   python -m pytest tests/
   ```
6. **Commit & Push**:
   ```bash
   git commit -m "feat: add amazing new feature"
   git push origin feature/amazing-new-feature
   ```
7. **Open a Pull Request** against the `main` branch.

---

## 🛠️ Code Style & Guidelines

- **Python**: Follow PEP 8 standards. Use type annotations for function signatures.
- **Documentation**: Write clear, docstrings for all public classes and methods.
- **Testing**: Ensure all unit tests pass with `pytest tests/`. Add new tests for added features.

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
