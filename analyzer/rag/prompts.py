"""Prompt templates for RAG.

Provides optimized prompt templates for different code analysis tasks.
"""

# Main code Q&A prompt
CODE_QA_PROMPT = """You are an expert code analyst. Answer the question based on the provided code context.

Think carefully about the code structure, patterns, and relationships. Provide accurate, helpful answers with specific references to the code.

Guidelines:
1. Be specific - reference actual function names, class names, and file locations
2. Include relevant code snippets when helpful
3. Explain the reasoning behind your analysis
4. If the context doesn't contain enough information, say so clearly
5. When discussing code quality or patterns, provide actionable suggestions

Context from codebase:
{context}

Question: {question}

Provide a clear, detailed answer:"""


# Code explanation prompt
CODE_EXPLANATION_PROMPT = """You are a code documentation expert. Explain the provided code clearly and thoroughly.

Context:
{context}

Explain this code, including:
1. **Purpose**: What does this code do?
2. **Key Components**: Main classes, functions, and their roles
3. **Design Patterns**: Any notable patterns or architectural decisions
4. **Dependencies**: Important imports and external dependencies
5. **Usage**: How would someone use this code?

Explanation:"""


# Code search prompt (for generating search-friendly queries)
CODE_SEARCH_PROMPT = """Given the user's question, generate search terms that would help find relevant code.

User question: {question}

Generate 3-5 search terms or phrases that would be useful for:
1. Finding relevant functions or methods
2. Finding relevant classes
3. Finding related concepts

Search terms:"""


# Summarization prompt
SUMMARIZATION_PROMPT = """Summarize the following code context concisely but thoroughly.

Context:
{context}

Provide a summary that includes:
1. Main purpose and functionality
2. Key components (classes, functions)
3. Notable patterns or approaches
4. Any potential issues or areas for improvement

Summary:"""


# Bug finding prompt
BUG_ANALYSIS_PROMPT = """Analyze the following code for potential bugs, issues, or areas of concern.

Context:
{context}

Question/Focus: {question}

Provide analysis covering:
1. **Potential Bugs**: Any logic errors, edge cases, or runtime issues
2. **Security Concerns**: Vulnerabilities or unsafe practices
3. **Code Quality**: Anti-patterns, code smells, or maintainability issues
4. **Recommendations**: Specific, actionable improvements

Analysis:"""


# Architecture analysis prompt
ARCHITECTURE_PROMPT = """Analyze the architecture and design of the following codebase.

Context:
{context}

Question: {question}

Provide architectural analysis including:
1. **Structure**: How is the code organized?
2. **Patterns**: What design patterns are used?
3. **Dependencies**: How do components interact?
4. **Quality**: Assessment of the overall design
5. **Suggestions**: Potential improvements

Analysis:"""


def get_prompt_template(prompt_type: str = "qa") -> str:
    """Get prompt template by type.
    
    Args:
        prompt_type: Type of prompt ("qa", "explain", "search", 
                     "summarize", "bugs", "architecture")
                     
    Returns:
        Prompt template string
    """
    templates = {
        "qa": CODE_QA_PROMPT,
        "explain": CODE_EXPLANATION_PROMPT,
        "search": CODE_SEARCH_PROMPT,
        "summarize": SUMMARIZATION_PROMPT,
        "bugs": BUG_ANALYSIS_PROMPT,
        "architecture": ARCHITECTURE_PROMPT,
    }
    return templates.get(prompt_type, CODE_QA_PROMPT)


def format_prompt(
    template: str,
    question: str = "",
    context: str = "",
    **kwargs
) -> str:
    """Format a prompt template with values.
    
    Args:
        template: Prompt template with placeholders
        question: User question
        context: Code context
        **kwargs: Additional template variables
        
    Returns:
        Formatted prompt string
    """
    return template.format(
        question=question,
        context=context,
        **kwargs
    )
