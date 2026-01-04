BASE_PROMPT_ALLSIDES = """
You are an expert in political media analysis. Your task is to determine the political bias of the following news text based on its language, tone, framing, and implicit political messages. Classify the text into one of the following categories:
A. Left
B. Center
C. Right

Your final output must consist of only one uppercase letter (A, B or C) corresponding to your classification. Do not include any explanation or extra text.
Read the content between "###Content starts." and "###Content ended.", and determine the political bias of the content.
"""

# BASE_PROMPT_ALLSIDES = """
# You are a hyper-rigorous political bias classification engine. Your exclusive function is to assign a political bias label to a news text segment strictly and solely by analyzing explicit, quantifiable linguistic features within the section marked “###Content starts.” and “###Content ended.” Absolutely disregard all external knowledge, author intent, context, or any inferred meaning.

# Apply this exhaustive, step-by-step analytical framework:

# 1. Decomposition and Tagging:
#    a. Break the text into sentences.
#    b. For each sentence, tag every instance of:
#       - Evaluative adjectives/adverbs directly describing political entities, actions, or policies.
#       - Verbs denoting approval, disapproval, endorsement, or criticism concerning political topics.
#       - Nouns or noun phrases acting as partisan identifiers, slogans, or loaded terminology associated with left, center, or right ideologies.
#       - Direct quotations or paraphrases explicitly supporting or denouncing any political position.

# 2. Quantitative Coding:
#    a. For each tagged instance, assign:
#       - Bias Direction: Left-favoring, Right-favoring, or Neutral/Balanced.
#       - Intensity Score: 1 (Mild/suggestive), 2 (Moderate/clear), 3 (Strong/forceful/dogmatic).
#       - Position Weight: Headline/title (4x), lead/first paragraph (3x), main body (2x), conclusion/summary (2x), supporting quote (1x).
#       - Framing Weight: Standalone assertion (2x), attribution to source (1x), hypothetical/conditional (0.5x).

#    b. Calculate the Weighted Bias Score for each instance: Intensity × Position Weight × Framing Weight.

# 3. Aggregation:
#    a. Sum Weighted Bias Scores for Left-favoring, Right-favoring, and Neutral/Balanced separately.
#    b. Compute each direction’s percentage of the total weighted score (sum Left, Right, Neutral).

# 4. Dominance Assessment:
#    a. If one direction’s percentage exceeds both others by at least 20 percentage points, and both opposing directions individually account for less than 25% of the total, assign that direction as dominant.
#    b. If none meets this threshold, or if Neutral/Balanced is highest but not dominant by threshold, default to Center.

# 5. Strict Output Protocol:
#    - Output ONLY a single uppercase letter according to the following mapping:
#      A: Left – Left-favoring cues are quantitatively and proportionally dominant by the criteria above.
#      B: Center – No direction meets dominance criteria, or Neutral/Balanced has the highest score.
#      C: Right – Right-favoring cues are quantitatively and proportionally dominant by the criteria above.
#    - Do NOT include any explanation, reasoning, punctuation, or additional text.

# Any ambiguous, implied, sarcastic, or context-dependent content MUST be excluded from consideration unless it contains unmistakable, explicit partisan markers. Your decision must be based exclusively on the systematic, mathematical analysis of explicit linguistic evidence within the provided boundaries.
# """

BASE_PROMPT_ALLSIDES_ORDER = """
You are an expert in political media analysis. Your task is to determine the political bias of the following news text based on its language, tone, framing, and implicit political messages. Classify the text into one of the following categories:
A. {bias_1}
B. {bias_2}
C. {bias_3}

Your final output must consist of only one uppercase letter (A, B or C) corresponding to your classification. Do not include any explanation or extra text.
Read the content between "###Content starts." and "###Content ended.", and determine the political bias of the content.
"""

BASE_PROMPT_SUMMARIZATION = """
Article:
{article}

Summarize the article in {summary_length} sentences.
"""

BASE_PROMPT_SUMMARIZATION_LEFT = """
Article:
{article}

Summarize the article in {summary_length} sentences.
Please emphasize a left-leaning perspective on the given article.
"""

BASE_PROMPT_SUMMARIZATION_CENTER = """
Article:
{article}

Summarize the article in {summary_length} sentences.
Please emphasize a neutral perspective on the given article.
"""


BASE_PROMPT_SUMMARIZATION_RIGHT = """
Article:
{article}

Summarize the article in {summary_length} sentences.
Please emphasize a right-leaning perspective on the given article.
"""

COT_PROMPT_ALLSIDES = """
You are an expert in political media analysis. Your task is to determine the political bias of the following news text based on its language, tone, framing, and implicit political messages. Classify the text into one of the following categories:
A. Left
B. Center
C. Right

You can reason step by step.
Read the content between "###Content starts." and "###Content ended.", and determine the political bias of the content.
"""