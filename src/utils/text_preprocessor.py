"""
text_preprocessor.py
─────────────────────
Preprocessing utilities for raw text, especially HTML content from Wikipedia.

The Natural Questions dataset contains raw Wikipedia HTML which needs to be
cleaned before chunking for better retrieval performance.
"""

import re
from typing import Optional


def preprocess_html_text(text: str, preserve_structure: bool = True) -> str:
    """
    Preprocess HTML-formatted text (e.g., from Wikipedia) for chunking.
    
    Converts HTML structure to markdown-style formatting that chunkers
    can recognize, and removes noisy HTML elements.
    
    Args:
        text: Raw text potentially containing HTML markup.
        preserve_structure: If True, convert HTML headings to markdown.
                           If False, strip all HTML tags completely.
    
    Returns:
        Cleaned text suitable for chunking.
    """
    if not text:
        return text
    
    # ─── Step 1: Convert HTML headings to markdown-style ─────────────────
    if preserve_structure:
        # <H1>Title</H1> → # Title
        text = re.sub(r'<[Hh]1[^>]*>\s*', '\n# ', text)
        text = re.sub(r'\s*</[Hh]1>', '\n', text)
        
        # <H2>Title</H2> → ## Title
        text = re.sub(r'<[Hh]2[^>]*>\s*', '\n## ', text)
        text = re.sub(r'\s*</[Hh]2>', '\n', text)
        
        # <H3>Title</H3> → ### Title
        text = re.sub(r'<[Hh]3[^>]*>\s*', '\n### ', text)
        text = re.sub(r'\s*</[Hh]3>', '\n', text)
        
        # <H4-H6> → #### (flatten deeper levels)
        text = re.sub(r'<[Hh][4-6][^>]*>\s*', '\n#### ', text)
        text = re.sub(r'\s*</[Hh][4-6]>', '\n', text)
    
    # ─── Step 2: Convert list items to markdown ──────────────────────────
    # <Li> → bullet point
    text = re.sub(r'<[Ll][Ii][^>]*>\s*', '\n• ', text)
    text = re.sub(r'\s*</[Ll][Ii]>', '', text)
    
    # ─── Step 3: Convert paragraphs to double newlines ───────────────────
    text = re.sub(r'<[Pp][^>]*>\s*', '\n\n', text)
    text = re.sub(r'\s*</[Pp]>', '\n\n', text)
    
    # ─── Step 4: Handle tables - extract text content ────────────────────
    # Table headers: <Th> → bold-style text
    text = re.sub(r'<[Tt][Hh][^>]*>\s*', ' **', text)
    text = re.sub(r'\s*</[Tt][Hh]>', '** ', text)
    
    # Table cells: <Td> → just space separation  
    text = re.sub(r'<[Tt][Dd][^>]*>\s*', ' ', text)
    text = re.sub(r'\s*</[Tt][Dd]>', ' ', text)
    
    # Table rows: add newline after each row
    text = re.sub(r'</[Tt][Rr]>', '\n', text)
    
    # ─── Step 5: Handle line breaks ──────────────────────────────────────
    text = re.sub(r'<[Bb][Rr]\s*/?>', '\n', text)
    text = re.sub(r'<[Hh][Rr]\s*/?>', '\n---\n', text)
    
    # ─── Step 6: Remove remaining HTML tags ──────────────────────────────
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # ─── Step 7: Clean up HTML entities ──────────────────────────────────
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
        '&mdash;': '—',
        '&ndash;': '–',
        '&hellip;': '...',
        '&copy;': '©',
        '&reg;': '®',
        '&trade;': '™',
        '&#x27;': "'",
        '&#x22;': '"',
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    # Handle numeric entities
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#[xX]([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
    
    # ─── Step 8: Normalize whitespace ────────────────────────────────────
    # Multiple spaces → single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Multiple newlines → max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace on lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Final trim
    text = text.strip()
    
    return text


def detect_html_content(text: str) -> bool:
    """
    Detect if text contains significant HTML markup.
    
    Args:
        text: Text to analyze.
    
    Returns:
        True if text appears to be HTML-formatted.
    """
    if not text:
        return False
    
    # Look for common HTML patterns
    html_patterns = [
        r'<[Hh][1-6][^>]*>',  # Headings
        r'<[Pp][^>]*>',       # Paragraphs
        r'<[Tt]able[^>]*>',   # Tables
        r'<[Uu][Ll][^>]*>',   # Unordered lists
        r'<[Oo][Ll][^>]*>',   # Ordered lists
        r'<[Dd]iv[^>]*>',     # Divs
    ]
    
    matches = sum(1 for pattern in html_patterns if re.search(pattern, text))
    return matches >= 2  # At least 2 different HTML elements


def preprocess_context(context: str, auto_detect: bool = True) -> str:
    """
    Smart preprocessing that auto-detects content type.
    
    Args:
        context: The context text to preprocess.
        auto_detect: If True, only preprocess if HTML is detected.
                    If False, always apply HTML preprocessing.
    
    Returns:
        Preprocessed text.
    """
    if auto_detect and not detect_html_content(context):
        return context
    
    return preprocess_html_text(context, preserve_structure=True)


if __name__ == "__main__":
    # Test with sample HTML
    sample = """<H1> Charles Bridge </H1> <P> Charles Bridge ( Czech : Karlův most ) is a historic bridge that crosses the Vltava river in Prague , Czech Republic . </P> <Table> <Tr> <Th> Coordinates </Th> <Td> 50 ° 5 ′ 11.21 '' N </Td> </Tr> </Table> <H2> History </H2> <P> The bridge was built in 1357 . </P> <Ul> <Li> First item </Li> <Li> Second item </Li> </Ul>"""
    
    print("=== Original HTML ===")
    print(sample[:200] + "...")
    print("\n=== Preprocessed ===")
    print(preprocess_html_text(sample))
    print("\n=== HTML Detected? ===", detect_html_content(sample))
