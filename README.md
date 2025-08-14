# Goal: To Create both and instruction following and human preference data set for Irish. This would be a first of its kind for the Irish language.

## Current Plan

Date: 14/08/2025
Project Plan: 
Context: I am developing an LLM for Irish. I have already continued pre-training on Irish data. I now want to create an Instruction Tuning data set and a human preference data set. I will use another LLM to generate prompt-response pairs. To prompt this LLM, I will provide reference text. I have text from Irish Wikipedia and Oireachtas debates.

Constraints:
- Report due 05/09/2025 so only 3 Weeks to finish, most models have timed usage limit rates.
- Cost of data synthesis

Key Questions of experiment:
1) Which LLM is best at generating Irish text?
	- This model will then be used for larger scale instruction-tuning and human feedback data synthesis.
2) Is Wikipedia or Oireachtas more useful as a source for reference text? 
	- This source will then be used to seed the reference text for the data synthesis.
	
Experiment Outline:

10 models / 2 refence sources => 5 main models flagship&cheaper, GPT, Claude, Gemini, Llama, Qwen.

instruction-tuning prompt template for the synthesising LLM:
TASK DESCRIPTION
You are given an Irish text source: {TEXT}
YOUR JOB:
Generate an instruction–response pair based on the provided text.
ALLOWED QUESTION TYPES
Is it true that ...
Explain ...
Describe ...
List the steps ...
REQUIREMENTS
The instruction must clearly incorporate the context from the provided text.
The response must be accurate and entirely in Irish.
Output only the instruction–response pair.
OUTPUT FORMAT
Instruction: <instruction in Irish>
Response: <response in Irish>


Plan Continued:
10 responses per dimension - 200 samples. 10 samples X 5 models X two price-points X two reference sources
A/B testing (gradio), will need to set up a gradio application to allow for easy annotation.
Test 1: 50/50 Source, vary model
Test 2: Fix best Model -> 40 samples 20/20


Annotators:
GPT-4 (LLM) -> will need to run this automatically
Me (Learner of Irish)
Native Speaker

- Compare ranking of (model,price,source)
	- Select highest ranked
	
- Compare alignment of LLM, Me and Native.
	- Can use this to justidy use of Me/LLM instead of Native for subsequent human feedback annotation.
	
