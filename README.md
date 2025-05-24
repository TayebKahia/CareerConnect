# CareerConnect 📌

**Integrated Career Guidance & Recruitment Platform**  


---

## 📖 Overview

CareerConnect is an AI-powered platform designed to address the key challenges faced by **job seekers** and **HR professionals**. It provides tools for resume analysis, job matching, interview preparation, and automated candidate screening — all through an intelligent, modular, and scalable system.

---

## 🚀 Problem Statements

### 🎯 For Job Seekers:
- Difficulty in crafting standout resumes.
- Limited interview preparation tools.
- Time-consuming job search and skill matching.

### 🎯 For HR Professionals:
- Manual and inefficient resume screening.
- Time-intensive candidate selection.

---

## 💡 Solution Features

### ✅ Job Seeker Portal:
- **Personalized Job Recommendations**

### ✅ HR Portal:
- **Automated Bulk Resume Screening**
- **Dashboard for Ranked Candidate Lists**


---


## 📈 Roadmap

- ✅ **MVP:** Job Seeker Portal (Resume Analysis, Job Matching, Interview Prep)
- 🟡 **Optional:** HR Portal (Bulk Screening, Dashboards)
- 🔄 Continuous model fine-tuning and feedback loop integration

---

## 🧪 Testing the Dual Ensemble Model

The Dual Ensemble model combines ML and NN approaches (without GNN) for more efficient job role prediction. Follow these steps to test it:

### 1. Installation Requirements

```bash
pip install -r requirements.txt
```

### 2. Test Prediction with Sample Text

```bash
python -m src.modeling.predict_job_role_dual --text "I am a software developer with experience in Python, JavaScript, and React. I have worked with MongoDB and AWS for cloud deployments."
```

### 3. Test with a File of Technologies

Create a text file with one technology per line:
```
Python
JavaScript
React
MongoDB
AWS
```

Then run:
```bash
python -m src.modeling.predict_job_role_dual --tech-file path/to/your/technologies.txt
```

### 4. Running Tests

```bash
python -m unittest tests.test_dual_ensemble
```

### 5. Training a New Model

To train a new dual ensemble model from scratch:

```bash
python -m src.modeling.train_dual_ensemble
```

---

## 📬 Contact

**Author:** Kahia Tayeb  
**Email:** t.kahia@esi-sba.dz  
**Location:** Blida, Algeria

---

> *Empowering careers with AI — one connection at a time.*

