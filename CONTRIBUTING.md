# Contributing to SkinVision

Thank you for your interest in contributing to **SkinVision**! 🎉

This project is an AI-powered educational and research showcase for dermoscopic skin lesion classification using deep learning. We welcome contributions, bug fixes, documentation improvements, and feature proposals.

---

## Code of Conduct & Medical Disclaimer

Please remember that **SkinVision is strictly a technical demonstration and portfolio project, NOT a medical device or diagnostic tool**.
- Any contribution or modification must not frame the software as clinically verified or approved for diagnostic use.
- The prominent medical disclaimers in the UI and documentation must remain intact.

---

## How to Contribute

### 1. Reporting Bugs
Before creating a new bug report, please check existing issues to ensure it hasn't already been reported. When reporting an issue, provide:
- A clear description of the bug.
- Step-by-step reproduction instructions.
- Relevant logs, error traces, and screenshots.
- Your OS, Python version, and GPU/CPU configuration.

### 2. Suggesting Features
Enhancement ideas are welcome! Please open an issue with:
- The motivation behind the feature.
- A proposed design or implementation outline.
- Any potential trade-offs or breaking changes.

---

## Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/cXOder6996/S-V.Revamped.git
   cd S-V.Revamped
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run Unit Tests**
   Make sure all tests pass before making any changes:
   ```bash
   pytest tests/
   ```

5. **Linting**
   Verify syntax and code style:
   ```bash
   flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,data
   ```

---

## Pull Request Guidelines

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Write Clean, Documented Code:**
   - Follow PEP 8 conventions.
   - Include docstrings and comments where appropriate.
   - If introducing new logic or modules, add unit tests under `tests/`.
3. **Verify Tests:**
   Ensure all tests pass locally:
   ```bash
   pytest tests/ -v
   ```
4. **Commit and Push:**
   Use clear, descriptive commit messages.
   ```bash
   git commit -m "feat: add support for XYZ"
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request:**
   Submit a PR against the `main` branch. Fill out the PR template with relevant context and link any related issues.

---

Thank you for helping improve SkinVision!
