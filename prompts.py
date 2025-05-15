is_heading_prompt = '''
You are a research assistant reading academic papers. 
Each paper has a section structure.

Your task: **Answer with ONLY one word: "Yes" or "No"** to the following question:  
**Is this title a main section heading (and not a sub-section)?**

Rules:
- Main section headings are usually numbered with a single number or numeral (e.g., "1.", "I.", "2.", "III.").
- Sub-sections usually have multiple levels of numbering (e.g., "2.1", "II.I", "3.2.1") or have a numbering that extends the main heading.
- If the title looks like it is under a main heading (i.e., part of a broader topic), answer "No".

<section structure>
{}
</section structure>

<title>
{}
</title>

Examples:
---
Section structure:
1. Introduction
2. Related Work
3. Methods
3.1 Preprocessing
3.2 Model Architecture
4. Results

Title: 2. Related Work  
Answer: Yes

Title: 3.1 Preprocessing  
Answer: No

---

Section structure:
I. Introduction
II. Background
II.I Previous Methods
III. Experiments

Title: II. Background  
Answer: Yes

Title: II.I Previous Methods  
Answer: No

---

Section structure:
1 Introduction
2 Related Work
2.1 Neural Networks
2.2 Transformers
3 Experiments

Title: 2.2 Transformers  
Answer: No

Title: 3 Experiments  
Answer: Yes

---

Now, based on the provided section structure and title, answer "Yes" if the title is a main heading or "No" if it is a subheading.
'''

summarize_abstract_prompt = '''
The following is an abstract from a 5G paper:

<abstract>
{}
</abstract>

summarize only the key technical content of this abstract. Ignore any other information present
'''

process_heading_prompt = '''
Based on the following paper structure describing the various section headings and subheadings: {}
and the summary of paper abstract: {}
summarize the following section and explain how it fits in the overall paper: {}
'''

process_subheading_prompt = '''
Based on the following paper structure: {}
and the summary of the abstract: {}
and the main heading summary: {}
summarize the following paper subsection and explain how it fits in the overall paper: {}
'''


inference_prompt_template = '''

You are a careful research assistant trained to extract technical terms from 5G security papers. You are given a section from a research paper about 4G, 5G, LTE, and OpenRAN (O-RAN) security. Your task is to carefully read the text and extract the names of any novel 4G, 5G, LTE, and OpenRAN (O-RAN) attacks mentioned in it.

- If an attack is described and it has a name or title, report the name exactly as it appears.
- Return only a list of attack names, with no extra explanation.
- If no attacks are mentioned, return an empty list: `[]`.

**Example**

Input:
"We first present our findings on the 5G NAS layer. 6.1.1 NAS Counter Reset. With this attack, the adversary exploits a vulnerability in the handling of NAS counters for message authentication codes (MACs)... (rest of text)"  


Output:
`["NAS Counter Reset"]`

---
Additional Notes:
- If multiple attacks are mentioned, list all of them in the order they appear.
- Do **not** include vulnerabilities, assumptions, or general descriptions — only named attacks.

Here is the section: 

'''

reranking_prompt =  """
You are an expert networking researcher specialized in 4G, 5G, LTE, and OpenRAN (O-RAN) security. Given a passage from a security  paper, score whether the passage contains names of novel 4G, 5G, LTE, and OpenRAN (O-RAN) attacks or vulnerabilities from 1 (irrelevant) to 10 (highly relevant). Rate the passage high only if it contains names of novel attacks. Rate it low if it contains related work, old attacks or other information.

Please report only the score within <score> tags.

Examples of how to report:

task: What is the capital of France?  
Passage: Paris is the capital and most populous city of France.  
<score> 10 </score>

task: What is the capital of France?  
Passage: France has a long history of culinary excellence and regional cuisine.  
<score> 2 </score>

task: How does photosynthesis work?  
Passage: Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.  
<score> 9 </score>

Now, score the following:

task: {} 
Passage: {} 
"""


retrieval_query = "You are an expert networking researcher specialized in 4G, 5G, LTE, OpenRAN (O-RAN) and security. Report the names of one or more attacks or vulnerabilites discovered against 4G, 5G, LTE, and OpenRAN presented in this paper"