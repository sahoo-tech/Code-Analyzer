

import re
import tokenize
from io import StringIO
from typing import Union

from analyzer.models.metrics import LOCMetrics
from analyzer.logging_config import get_logger

logger = get_logger("metrics.loc")


class LOCCalculator:

    
    def calculate(self, code: str) -> LOCMetrics:
 
        lines = code.splitlines()
        total = len(lines)
        
        if total == 0:
            return LOCMetrics()
        
        # Count blank lines
        blank = sum(1 for line in lines if not line.strip())
        
        # Use tokenizer for accurate comment and docstring detection
        comment_lines = set()
        docstring_lines = set()
        
        try:
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))
            
            for i, token in enumerate(tokens):
                tok_type = token.type
                tok_string = token.string
                start_line = token.start[0]
                end_line = token.end[0]
                
                if tok_type == tokenize.COMMENT:
                    comment_lines.add(start_line)
                
                elif tok_type == tokenize.STRING:
                    # Check if this is a docstring
                    if self._is_docstring(tokens, i):
                        for line_no in range(start_line, end_line + 1):
                            docstring_lines.add(line_no)
        
        except tokenize.TokenizeError:
            # Fallback to regex-based counting
            return self._fallback_calculate(code)
        
        comments = len(comment_lines)
        docstrings = len(docstring_lines)
        
        # Source lines = total - blank - pure comment lines
        # (Lines with code and comments count as source)
        pure_comment_lines = sum(
            1 for i, line in enumerate(lines, 1)
            if i in comment_lines and line.strip().startswith('#')
        )
        
        source = total - blank - pure_comment_lines
        
        return LOCMetrics(
            total=total,
            source=source,
            comments=comments,
            blank=blank,
            docstrings=docstrings,
        )
    
    def _is_docstring(self, tokens: list, index: int) -> bool:
 
        if index == 0:
            return True  # Module docstring
        
        # Look back for def/class
        for i in range(index - 1, -1, -1):
            prev_token = tokens[i]
            
            # Skip newlines and indentation
            if prev_token.type in (tokenize.NEWLINE, tokenize.INDENT, 
                                   tokenize.NL, tokenize.ENCODING):
                continue
            
            # If we hit a colon, this might be a docstring
            if prev_token.type == tokenize.OP and prev_token.string == ':':
                return True
            
            # If we hit anything else, it's not a docstring
            break
        
        return False
    
    def _fallback_calculate(self, code: str) -> LOCMetrics:
        """Fallback calculation using regex."""
        lines = code.splitlines()
        total = len(lines)
        blank = 0
        comments = 0
        
        in_multiline_string = False
        multiline_char = None
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                blank += 1
                continue
            
            # Check for multi-line strings (rough approximation)
            if not in_multiline_string:
                if stripped.startswith('#'):
                    comments += 1
                elif '"""' in stripped or "'''" in stripped:
                    # Could be docstring
                    quote = '"""' if '"""' in stripped else "'''"
                    count = stripped.count(quote)
                    if count == 1:
                        in_multiline_string = True
                        multiline_char = quote
            else:
                if multiline_char in stripped:
                    in_multiline_string = False
                    multiline_char = None
        
        source = total - blank - comments
        
        return LOCMetrics(
            total=total,
            source=source,
            comments=comments,
            blank=blank,
            docstrings=0,  # Can't accurately count with fallback
        )


def calculate_loc(code: str) -> LOCMetrics:

    return LOCCalculator().calculate(code)
