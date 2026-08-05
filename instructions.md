# LLMs grading LLMs—with receipts

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
You're a consultant tasked with designing and implementing an evaluation strategy for a client's LLM use case. This lab takes you through the complete process: from auditing existing benchmarks to designing your own evaluation, and finally implementing and running it in Python.

**Learning Objectives**
- [ ] Analyze and critique existing benchmarks for relevance, contamination, and saturation risks
- [ ] Design custom evaluation prompts with clear ground truth and verification methods
- [ ] Create LLM-as-judge evaluation prompts following best practices
- [ ] Implement a complete evaluation pipeline using OpenAI/LangChain
- [ ] Run evaluations and analyze results including accuracy, cost, and time metrics
- [ ] Write professional evaluation reports with appropriate hedging and caveats

**Estimated Time:** 240 minutes (4 hours)

**Prerequisites:**
- [ ] Understanding of LLM evaluation concepts (from lesson content)
- [ ] Familiarity with OpenAI API or LangChain
- [ ] Basic Python programming experience
- [ ] Understanding of benchmark categories (reasoning, knowledge, code, instruction following, tool calling)

---

## Introduction

This lab simulates a real-world consulting scenario where you must:
1. **Audit** existing benchmarks for a client's use case
2. **Design** a custom evaluation strategy from scratch
3. **Implement** and run the evaluation using Python
4. **Report** findings with appropriate professional caveats

**Why this matters:**
- Real evaluations require understanding benchmark limitations
- Custom evaluations are often necessary for specific use cases
- Implementation skills are essential for production systems
- Professional reporting requires understanding evaluation caveats

**Success criteria:**
- [ ] Complete benchmark audit with 3 evaluated benchmarks
- [ ] Design 5 custom evaluation prompts with verification strategies
- [ ] Implement working LLM-as-judge evaluation in Python
- [ ] Generate evaluation report with metrics and recommendations

---

## Reference Materials

**Documentation:**
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Evaluation Documentation](https://python.langchain.com/docs/guides/evaluation/)
- [LangChain LLM Judge Example](https://python.langchain.com/docs/guides/evaluation/string/custom_criteria/)

**Benchmark Resources:**
- [Papers with Code - LLM Benchmarks](https://paperswithcode.com/task/language-modelling)
- [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/)

---

## Part 1: Benchmark Audit & Evaluation Design (180 minutes)

### Hour 1 — Client Scenario & Benchmark Audit (60 min)

#### Step 1 — Pick Your Client Scenario (10 min)

**Objective:** Choose a realistic consulting context for your evaluation.

**What to do:**
Choose one fictional but realistic consulting context from the examples below, or create your own:

**Option A: Financial Services**
- A bank wants to automate loan rejection letters
- Requirements: Must be clear, compliant, and empathetic
- Key concerns: Accuracy, tone, regulatory compliance

**Option B: Healthcare**
- A hospital wants to summarize patient records
- Requirements: Must preserve critical medical information
- Key concerns: Accuracy, completeness, medical terminology

**Option C: Retail**
- A retailer wants a customer service chatbot
- Requirements: Must handle returns, product questions, complaints
- Key concerns: Helpfulness, tone, resolution rate

**Option D: Your Own Scenario**
- Create a realistic scenario with clear requirements and concerns

**Deliverable:** Write 2-3 sentences describing your chosen scenario, including:
- The client's goal
- Key requirements
- Main concerns or failure modes

---

#### Step 2 — Find & Critique 3 Existing Benchmarks (50 min)

**Objective:** Research and evaluate existing benchmarks for relevance to your scenario.

**What to do:**
Using the benchmark categories from the lesson (reasoning, knowledge, code, instruction following, tool calling, etc.), find 3 benchmarks that seem relevant to your scenario.

For each benchmark, fill in the following evaluation card:

**Benchmark Evaluation Card Template:**

```
Benchmark Name: [Name]
Year: [Year]
Source: [URL or paper citation]

Why it seemed relevant:
[2-3 sentences explaining why this benchmark relates to your scenario]

Contamination risk:
[ ] Low - Model likely not trained on this data
[ ] Medium - Some overlap possible
[ ] High - Model definitely saw this during training
[Explanation: ...]

Saturation risk:
[ ] Low - Benchmark is challenging
[ ] Medium - Some models perform well
[ ] High - Many models achieve near-perfect scores
[Explanation: ...]

Format:
[ ] Multiple Choice
[ ] Free-form text
[ ] Code generation
[ ] Other: [specify]

Verdict:
[ ] Use it as-is
[ ] Adapt it (explain how)
[ ] Reject it (explain why)
```

**Where to find benchmarks:**
- Papers with Code: https://paperswithcode.com/task/language-modelling
- Hugging Face Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- HELM: https://crfm.stanford.edu/helm/
- Academic papers (Google Scholar, arXiv)

**Deliverable:** Complete 3 benchmark evaluation cards. Save these for your final report.

---

### Hour 2 — Design Your Own Mini Benchmark (120 min)

#### Step 3 — Write 5 Evaluation Prompts (60 min)

**Objective:** Create custom evaluation prompts tailored to your client scenario.

**What to do:**
Write 5 prompts you would actually test a model on for your scenario. For each prompt, decide:

1. **The prompt itself** - What would you ask the model?
2. **Ground truth** - Is there a correct answer? If so, what is it?
3. **Verification method** - How would you verify the answer?
   - Rule-based (exact match, regex, keyword check)
   - Human evaluation (subjective quality)
   - LLM-as-judge (automated quality assessment)
4. **Failure mode** - What are you most worried about?
   - Hallucination
   - Incorrect tone
   - Missing information
   - Safety/toxicity
   - Other: [specify]

**Evaluation Prompt Template:**

```
Prompt #1: [Title]

Prompt:
[The actual prompt/question you would test]

Ground Truth:
[ ] Yes - [Describe the correct answer]
[ ] No - [Explain why there's no single correct answer]

Verification Method:
[ ] Rule-based: [Describe the rules/checks]
[ ] Human evaluation: [Describe what humans would assess]
[ ] LLM-as-judge: [Describe what the judge would evaluate]

Primary Failure Mode:
[What you're most concerned about with this prompt]

Why this prompt matters:
[1-2 sentences on why this evaluation is important for your scenario]
```

**Tips:**
- Make prompts realistic (similar to what the model would see in production)
- Vary the difficulty level (some easy, some challenging)
- Include edge cases or failure scenarios
- Consider different aspects: accuracy, tone, completeness, safety

**Deliverable:** Complete 5 evaluation prompt cards.

---

#### Step 4 — Design Your Judge (60 min)

**Objective:** Create a complete LLM-as-judge prompt following best practices.

**What to do:**
Pick one of your 5 prompts and write a full LLM-as-judge evaluation prompt following the anatomy from the lesson:

1. **Task Description** - What is the model being asked to do?
2. **Evaluation Criteria** - What makes a response good or bad?
3. **Reasoning Steps** - How should the judge think through the evaluation?
4. **Output Format** - What format should the judge use? (e.g., JSON with score and reasoning)

**Judge Prompt Template:**

```
Task Description:
[Describe the original task the model was asked to perform]

Evaluation Criteria:
1. [Criterion 1]: [Description]
2. [Criterion 2]: [Description]
3. [Criterion 3]: [Description]

Reasoning Steps:
Step 1: [What should the judge check first?]
Step 2: [What should the judge check second?]
Step 3: [What should the judge check third?]

Output Format:
{
  "score": [1-5 or 0-1],
  "reasoning": "[Explanation of the score]",
  "criteria_met": {
    "criterion_1": true/false,
    "criterion_2": true/false,
    "criterion_3": true/false
  }
}
```

**Bias Analysis:**
After writing your judge prompt, answer these questions:

1. **Hidden biases:** What hidden biases might your judge have?
   - Cultural assumptions?
   - Language preferences?
   - Style biases?
   - Domain-specific assumptions?

2. **Calibration:** How would you calibrate your judge?
   - What reference examples would you use?
   - How would you handle edge cases?
   - What would you do if the judge is too strict/lenient?

**Deliverable:** 
- Complete judge prompt with all 4 components
- Bias analysis (2-3 paragraphs)
- Calibration strategy (2-3 paragraphs)

---

### Hour 3 — Evaluation Report Card (60 min)

#### Step 5 — Write a 1-Page "Evaluation Memo" to Your Client (40 min)

**Objective:** Practice professional evaluation reporting with appropriate caveats.

**What to do:**
Pretending you've run the evaluation, write a 1-page memo to your client. The memo must include:

1. **Executive Summary** (2-3 sentences)
   - What you measured and why

2. **Methodology** (2-3 paragraphs)
   - Benchmarks used (or custom evaluation)
   - Evaluation approach
   - Models tested

3. **Results** (2-3 paragraphs)
   - Key metrics (accuracy, performance)
   - Model comparison (if applicable)
   - Notable findings

4. **Caveats & Limitations** (1-2 paragraphs)
   - What you cannot guarantee
   - Reference saturation/contamination/reproducibility concerns
   - Scope limitations

5. **Recommendation** (1 paragraph)
   - Model A vs Model B (or single model recommendation)
   - Use hedging language: "under these conditions, for this task..."
   - Confidence level

6. **Additional Metrics** (1 paragraph)
   - Beyond accuracy: time, token consumption, environmental cost
   - Cost-benefit analysis if relevant

**Memo Template:**

```
TO: [Client Name]
FROM: [Your Name]
DATE: [Date]
SUBJECT: LLM Evaluation Results - [Your Scenario]

EXECUTIVE SUMMARY
[2-3 sentences]

METHODOLOGY
[2-3 paragraphs]

RESULTS
[2-3 paragraphs]

CAVEATS & LIMITATIONS
[1-2 paragraphs]

RECOMMENDATION
[1 paragraph]

ADDITIONAL METRICS
[1 paragraph]
```

**Deliverable:** 1-page evaluation memo (approximately 400-500 words).

---

#### Step 6 — Reflection (20 min)

**Objective:** Think critically about evaluation design challenges.

**What to do:**
Answer these three questions in writing (2-3 paragraphs each):

**Question 1:** What would change in your evaluation design if your client's data was in French?
- How would you adapt benchmarks?
- What new challenges would arise?
- How would you verify quality in a non-English context?

**Question 2:** Your client asks "is this model AGI-level?" — how do you respond?
- What does "AGI-level" mean in this context?
- What evaluations would be needed to answer this?
- What caveats would you include?

**Question 3:** What is the one thing you could not evaluate without a human, and why?
- What aspects require human judgment?
- Why can't LLM-as-judge or rule-based methods work?
- How would you incorporate human evaluation in practice?

**Deliverable:** Written responses to all three reflection questions (approximately 300-400 words total).

---

## Part 2: Implementation & Execution (60 minutes)

### Hour 4 — Implement & Run Your Evaluation (60 min)

**Objective:** Implement your LLM-as-judge evaluation in Python and run it on real examples.

**What to do:**
Now you'll implement the evaluation you designed in Hours 1-3. You can use either:
- **OpenAI API directly** (simpler, more control)
- **LangChain** (more structure, built-in evaluation helpers)

Choose the approach that works best for you. Both rely on the **OpenAI API**, which is billed per use; use a small model such as `gpt-4o-mini` to keep costs low while you develop.

---

#### Step 7: Setup & Environment (10 min)

**Objective:** Set up your Python environment and install dependencies.

**What to do:**

1. Create a new Python file: `llm_judge_evaluation.py` or use a Jupyter notebook

2. Install required packages: `openai`, `langchain`, `langchain-openai`, `python-dotenv`

3. Set up your OpenAI API key:
   - Create a `.env` file in your project directory
   - Add: `OPENAI_API_KEY=your_key_here`
   - Make sure `.env` is in your `.gitignore`

4. Set up your environment:
   - Import necessary libraries (OpenAI client or LangChain)
   - Load environment variables
   - Initialize your LLM client (use `gpt-4o-mini` for free tier friendly option)

---

#### Step 8: Implement Your Evaluation Prompt (15 min)

**Objective:** Code your LLM-as-judge prompt from Step 4.

**What to do:**
Implement the judge prompt you designed in Step 4. Create a function that takes:
- The original prompt (what you asked the model)
- The model's response
- Returns: score, reasoning, and criteria assessment

**Implementation guidance:**
- Use the judge prompt structure you designed in Step 4
- Include the task description, evaluation criteria, reasoning steps, and output format
- Make sure the judge returns structured output (JSON format recommended)
- Test your function with a simple example before running on all test cases

---

#### Step 9: Create Test Dataset (10 min)

**Objective:** Prepare test cases for your evaluation.

**What to do:**
Create a small test dataset (3-5 examples) based on your 5 prompts from Step 3. For each prompt, you'll need:
- The prompt itself
- A model response (you can generate this using the model, or use a pre-written sample)
- Ground truth (if applicable)
- Expected criteria that should be met

**Structure your test cases:**
- Create a list or dictionary of test cases
- Include an ID, the prompt, ground truth (if available), and expected criteria
- Generate responses for each prompt using your chosen model

---

#### Step 10: Run Evaluation & Collect Metrics (20 min)

**Objective:** Run your evaluation on all test cases and collect metrics.

**What to do:**
1. Run the evaluation on all test cases using your judge function
2. Collect metrics: scores, time, token usage, cost
3. Calculate aggregate statistics (average score, min/max scores, total time, etc.)
4. Save results to a JSON file for later analysis

**Metrics to track:**
- Individual scores for each test case
- Reasoning provided by the judge
- Criteria met for each case
- Time taken per evaluation
- Total time for all evaluations
- Token usage (if available from API)
- Estimated cost (optional but recommended)

**Output format:**
- Save results in a structured format (JSON recommended)
- Include both individual results and aggregate statistics

---

#### Step 11: Analyze & Visualize Results (5 min)

**Objective:** Create a simple summary of your results.

**What to do:**
Create a summary report showing:
- Score distribution across all test cases
- Criteria performance (how many criteria were met per case)
- Time and cost metrics
- Key insights from the judge's reasoning

**Analysis to include:**
- Print or display detailed results for each test case
- Show score, time taken, and criteria met for each case
- Summarize the judge's reasoning (you may want to truncate long reasoning text)
- Identify patterns or issues across test cases

---

## Deliverables

### Submission hygiene

- **Filenames:** Use clear, descriptive names (avoid vague names such as `lab.ipynb`, `final_v2.py`, or `untitled.md`).
- **Scope:** Your **GitHub** repository must contain **only materials for this lab**—no unrelated projects, dumps, or personal files.
- **README:** Include a `README.md` that briefly explains what each main file or folder is for (a short map of your file structure).

**GitHub only:** Submit the URL to a **GitHub repository** that contains everything for this lab (Markdown, code, exports, images, decks). Do **not** submit a standalone Google Doc, Notion page, or cloud-only link as your primary deliverable—put sources or exports (for example `.md`, `.pdf`, `.pptx`, screenshots) **in the repository**.

Submit the following items:

1. **Benchmark Audit (Step 2)**
   - 3 completed benchmark evaluation cards
   - Saved as: `benchmark_audit.md` or `benchmark_audit.json`

2. **Evaluation Design (Steps 3-4)**
   - 5 evaluation prompt cards
   - 1 complete LLM-as-judge prompt with bias analysis
   - Saved as: `evaluation_design.md`

3. **Evaluation Memo (Step 5)**
   - 1-page client memo
   - Saved as: `evaluation_memo.md`

4. **Reflection (Step 6)**
   - Answers to 3 reflection questions
   - Saved as: `reflection.md`

5. **Implementation (Steps 7-11)**
   - Python code file: `llm_judge_evaluation.py` (or `.ipynb`)
   - Evaluation results: `evaluation_results.json`
   - Brief summary: `implementation_summary.md` (2-3 paragraphs describing what you built and key findings)

**Submission Format:**
- Create a folder: `lab_llm_judges_[your_name]`
- Include all deliverables above
- Add a `README.md` with:
  - Your chosen scenario
  - Brief overview of your approach
  - How to run your code (if applicable)

---

## Success Criteria

Your lab is complete when:
- [ ] You've audited 3 benchmarks with complete evaluation cards
- [ ] You've designed 5 custom evaluation prompts with verification methods
- [ ] You've created a complete LLM-as-judge prompt with bias analysis
- [ ] You've written a professional evaluation memo with appropriate caveats
- [ ] You've reflected on evaluation design challenges
- [ ] You've implemented a working evaluation in Python
- [ ] You've run evaluations and collected metrics (scores, time, cost)
- [ ] All deliverables are submitted in the required format

---

## Tips & Troubleshooting

**Common Issues:**

1. **JSON parsing errors in judge responses**
   - Use `response_format={"type": "json_object"}` in OpenAI API
   - Add error handling: `try/except` around JSON parsing
   - Consider asking for JSON in a code block if needed

2. **Inconsistent judge scores**
   - This is normal! LLM judges have variance
   - Consider running multiple times and averaging
   - Use lower temperature (0) for more consistent results

3. **High API costs**
   - Use `gpt-4o-mini` instead of `gpt-4` or `gpt-4-turbo`
   - Cache responses when possible
   - Start with a small test set (3-5 examples)

4. **Judge is too strict/lenient**
   - Add calibration examples to your judge prompt
   - Provide score anchors (examples of what each score means)
   - Consider using a rubric with specific criteria

**Best Practices:**
- Test your judge on known good/bad examples first
- Save all API responses for later analysis
- Document any assumptions or limitations
- Consider running evaluations multiple times to check consistency

---

## Extension Activities (Optional)

If you finish early or want to go deeper:

1. **Compare two models:** Evaluate the same prompts on two different models and compare results
2. **Human evaluation:** Have a classmate evaluate 2-3 examples and compare with your LLM judge
3. **Calibration study:** Test your judge on 10 examples with known quality levels and see if scores correlate
4. **Cost optimization:** Experiment with different models for the judge (e.g., gpt-3.5-turbo vs gpt-4o-mini) and compare cost/quality trade-offs

---

*Good luck! Remember: evaluation is as much art as science. The goal is to build something useful, not perfect.*
