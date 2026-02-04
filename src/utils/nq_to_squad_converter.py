"""
Natural Questions to SQuAD Format Converter
===========================================

Converts Google's Natural Questions dataset to SQuAD-compatible format.

Based on actual NQ structure:
- document['tokens'] is a DICT with keys: 'token', 'is_html', 'start_byte', 'end_byte'
- The 'token' key contains a LIST of all token strings
- annotations['short_answers'] is a LIST of dicts (one per annotator)
- Each annotator dict has 'text' field which is a list of answer strings

Author: ML Engineering Team
Version: 2.2.0
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package - {e.name}")
    print("\nInstall dependencies with: pip install datasets tqdm")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration parameters for NQ to SQuAD conversion."""
    
    # Text extraction
    min_answer_length: int = 4
    min_context_length: int = 50
    max_context_length: int = 100000  # Increased to handle NQ's long Wikipedia articles
    
    # Dataset
    dataset_name: str = "google-research-datasets/natural_questions"
    dataset_config: str = "dev"
    split: str = "validation"
    
    # Output
    default_output_path: str = "data/natural_questions_squad.json"
    
    # Logging
    log_level: str = "INFO"


@dataclass
class ConversionStats:
    """Statistics tracked during conversion."""
    
    total_processed: int = 0
    successfully_converted: int = 0
    skipped_no_question: int = 0
    skipped_no_document_tokens: int = 0
    skipped_no_annotations: int = 0
    skipped_no_valid_answer: int = 0
    skipped_context_too_short: int = 0
    skipped_context_too_long: int = 0
    skipped_answer_not_in_context: int = 0
    skipped_answer_too_short: int = 0
    
    context_lengths: List[int] = field(default_factory=list)
    answer_lengths: List[int] = field(default_factory=list)
    
    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_no_question + 
            self.skipped_no_document_tokens + 
            self.skipped_no_annotations + 
            self.skipped_no_valid_answer + 
            self.skipped_context_too_short + 
            self.skipped_context_too_long + 
            self.skipped_answer_not_in_context +
            self.skipped_answer_too_short
        )
    
    @property
    def conversion_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.successfully_converted / self.total_processed
    
    @property
    def avg_context_length(self) -> float:
        if not self.context_lengths:
            return 0.0
        return sum(self.context_lengths) / len(self.context_lengths)
    
    @property
    def avg_answer_length(self) -> float:
        if not self.answer_lengths:
            return 0.0
        return sum(self.answer_lengths) / len(self.answer_lengths)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging with appropriate format and level."""
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# Text Extraction
# =============================================================================

class DocumentProcessor:
    """Handles extraction of clean text from NQ document tokens."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def extract_context_from_tokens(self, document: Dict) -> str:
        """
        Extract clean context text from document tokens.
        
        NQ document['tokens'] is a DICT with keys: 'token', 'is_html', 'start_byte', 'end_byte'
        The 'token' key contains a LIST of all tokens.
        
        Args:
            document: NQ document dict with 'tokens' field
            
        Returns:
            Clean context text string
        """
        tokens_data = document.get('tokens')
        
        if not tokens_data:
            self.logger.debug("No tokens found in document")
            return ""
        
        # Extract token list
        tokens = []
        
        if isinstance(tokens_data, dict):
            # NQ structure: tokens_data is dict with 'token' key containing list
            if 'token' in tokens_data:
                token_list = tokens_data['token']
                if isinstance(token_list, list):
                    tokens = [str(t) for t in token_list]
                    self.logger.debug(f"Extracted {len(tokens)} tokens from 'token' key")
                else:
                    self.logger.error(f"Expected list for 'token' key, got {type(token_list)}")
                    return ""
            else:
                self.logger.error(f"'token' key not found. Available keys: {list(tokens_data.keys())}")
                return ""
                
        elif isinstance(tokens_data, list):
            # Fallback: if tokens_data is already a list
            tokens = [str(t) for t in tokens_data]
            self.logger.debug(f"Using {len(tokens)} tokens from list")
        else:
            self.logger.warning(f"Unexpected tokens type: {type(tokens_data)}")
            return ""
        
        if not tokens:
            self.logger.error("Token list is empty after extraction")
            return ""
        
        # Verify we have actual text tokens
        self.logger.debug(f"First 3 tokens: {tokens[:3]}")
        
        # Join tokens with spaces
        raw_text = " ".join(tokens)
        self.logger.debug(f"Raw text length: {len(raw_text)} chars")
        
        # Clean the text
        clean_text = self._clean_text(raw_text)
        self.logger.debug(f"Clean text length: {len(clean_text)} chars")
        
        return clean_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with minimal removal."""
        
        # Unescape HTML entities
        text = unescape(text)
        
        # Only remove the most obvious Wikipedia UI noise
        # Be very conservative - we want to keep article content
        patterns_to_remove = [
            r'Jump to:?\s*navigation,?\s*search',
            r'\[\s*edit\s*\]',
            r'\[\s*edit source\s*\]',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


# =============================================================================
# Answer Extraction
# =============================================================================

class AnswerExtractor:
    """Handles extraction of answers from NQ annotations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract_best_answer(self, annotations: Dict) -> Optional[str]:
        """
        Extract the best answer from annotations.
        
        annotations['short_answers'] is a LIST of dicts (one per annotator).
        Each dict has 'text' field which is a list of answer strings.
        
        Args:
            annotations: NQ annotations dict
            
        Returns:
            Answer text or None
        """
        if not annotations:
            return None
        
        short_answers_list = annotations.get('short_answers', [])
        
        if not short_answers_list:
            self.logger.debug("No short_answers in annotations")
            return None
        
        # Try each annotator's response
        for annotator_answers in short_answers_list:
            if not isinstance(annotator_answers, dict):
                continue
            
            answer_texts = annotator_answers.get('text', [])
            
            # Return first non-empty answer
            if answer_texts and len(answer_texts) > 0:
                answer = answer_texts[0]
                if answer and answer.strip():
                    return answer.strip()
        
        self.logger.debug("No valid answer text found in any annotation")
        return None


# =============================================================================
# Converter
# =============================================================================

class NQToSQuADConverter:
    """Main converter class for NQ to SQuAD format."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.document_processor = DocumentProcessor(config, logger)
        self.answer_extractor = AnswerExtractor(logger)
        self.stats = ConversionStats()
    
    def convert_example(self, example: Dict) -> Optional[Dict]:
        """
        Convert single NQ example to SQuAD format.
        
        Args:
            example: NQ example dict
                
        Returns:
            SQuAD format dict or None if conversion fails
        """
        self.stats.total_processed += 1
        
        # Extract question
        question_text = self._extract_question(example)
        if not question_text:
            self.stats.skipped_no_question += 1
            return None
        
        # Extract context
        context_text = self._extract_context(example)
        if not context_text:
            self.stats.skipped_no_document_tokens += 1
            return None
        
        # Validate context length
        if not self._validate_context_length(context_text):
            return None
        
        # Extract answer
        if 'annotations' not in example:
            self.stats.skipped_no_annotations += 1
            return None
        
        answer_text = self._extract_answer(example['annotations'])
        if not answer_text:
            self.stats.skipped_no_valid_answer += 1
            return None
        
        # Validate answer
        if not self._validate_answer(answer_text, context_text):
            return None
        
        # Build SQuAD format
        squad_example = self._build_squad_example(
            example_id=str(example.get('id', '')),
            question=question_text,
            context=context_text,
            answer=answer_text
        )
        
        # Track statistics
        self.stats.successfully_converted += 1
        self.stats.context_lengths.append(len(context_text))
        self.stats.answer_lengths.append(len(answer_text))
        
        return squad_example
    
    def _extract_question(self, example: Dict) -> Optional[str]:
        """Extract question text from example."""
        question = example.get('question', {})
        if isinstance(question, dict):
            question_text = question.get('text', '').strip()
        else:
            question_text = str(question).strip()
        
        return question_text if question_text else None
    
    def _extract_context(self, example: Dict) -> Optional[str]:
        """Extract context text from document."""
        document = example.get('document', {})
        if not document:
            return None
        
        context_text = self.document_processor.extract_context_from_tokens(document)
        return context_text if context_text else None
    
    def _validate_context_length(self, context: str) -> bool:
        """Validate context length against configured bounds."""
        length = len(context)
        
        if length < self.config.min_context_length:
            self.stats.skipped_context_too_short += 1
            if self.stats.skipped_context_too_short <= 3:
                self.logger.warning(f"Context too short: {length} chars (min: {self.config.min_context_length})")
            return False
        
        if length > self.config.max_context_length:
            self.stats.skipped_context_too_long += 1
            return False
        
        return True
    
    def _extract_answer(self, annotations: Dict) -> Optional[str]:
        """Extract answer from annotations."""
        return self.answer_extractor.extract_best_answer(annotations)
    
    def _validate_answer(self, answer: str, context: str) -> bool:
        """Validate answer against requirements."""
        
        # Length check
        if len(answer) < self.config.min_answer_length:
            self.stats.skipped_answer_too_short += 1
            return False
        
        # Answer must appear in context (case-insensitive)
        if answer.lower() not in context.lower():
            self.stats.skipped_answer_not_in_context += 1
            return False
        
        return True
    
    def _build_squad_example(
        self, 
        example_id: str, 
        question: str, 
        context: str, 
        answer: str
    ) -> Dict:
        """Build SQuAD format dictionary."""
        
        # Validate that context is actually a string
        if not isinstance(context, str):
            self.logger.warning(f"Context is not a string for example {example_id}: {type(context)}")
            context = str(context)
        
        return {
            'id': example_id,
            'question': question,
            'context': context,
            'answer': answer,
            'metadata': {
                'source': 'natural_questions',
                'original_id': example_id,
                'context_length': len(context),
                'answer_length': len(answer),
            }
        }


# =============================================================================
# Pipeline
# =============================================================================

class ConversionPipeline:
    """Orchestrates the full conversion pipeline."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.converter = NQToSQuADConverter(config, logger)
    
    def run(
        self, 
        sample_size: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Tuple[List[Dict], ConversionStats]:
        """
        Run the full conversion pipeline.
        
        Args:
            sample_size: Number of examples to process (None = all)
            output_path: Output file path
            
        Returns:
            Tuple of (converted examples, statistics)
        """
        self.logger.info("="*70)
        self.logger.info("Natural Questions to SQuAD Converter - Starting")
        self.logger.info("="*70)
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Apply sampling if requested
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
            self.logger.info(f"Processing sample of {len(dataset):,} examples")
        else:
            self.logger.info(f"Processing all {len(dataset):,} examples")
        
        # Convert examples
        converted_examples = self._convert_dataset(dataset)
        
        # Save results
        output_file = output_path or self.config.default_output_path
        self._save_results(converted_examples, output_file)
        
        # Print summary
        self._print_summary(output_file)
        
        return converted_examples, self.converter.stats
    
    def _load_dataset(self):
        """Load Natural Questions dataset with proper config and split."""
        self.logger.info(f"Loading dataset: {self.config.dataset_name}")
        self.logger.info(f"  Config: {self.config.dataset_config}")
        self.logger.info(f"  Split: {self.config.split}")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split
            )
            
            self.logger.info(f"✓ Successfully loaded {len(dataset):,} examples")
            
            if len(dataset) > 0:
                example = dataset[0]
                self.logger.info(f"  Dataset columns: {list(example.keys())}")
            
            return dataset
        
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _convert_dataset(self, dataset) -> List[Dict]:
        """Convert all examples in dataset."""
        self.logger.info("Converting examples to SQuAD format...")
        self.logger.info(f"  Filters: min_answer={self.config.min_answer_length}, "
                        f"context={self.config.min_context_length}-{self.config.max_context_length}")
        
        converted_examples = []
        
        for example in tqdm(dataset, desc="Converting", unit="ex"):
            squad_example = self.converter.convert_example(example)
            
            if squad_example:
                converted_examples.append(squad_example)
        
        self.logger.info(f"✓ Conversion complete: {len(converted_examples):,} examples")
        
        return converted_examples
    
    def _save_results(self, examples: List[Dict], output_path: str):
        """Save converted examples to JSON file."""
        self.logger.info(f"Saving results to: {output_path}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.converter.stats
        
        output_data = {
            'examples': examples,
            'metadata': {
                'source': 'natural_questions',
                'dataset_config': self.config.dataset_config,
                'split': self.config.split,
                'version': '2.1.0',
                'total_examples_processed': stats.total_processed,
                'converted_examples': stats.successfully_converted,
                'skipped_examples': stats.total_skipped,
                'conversion_rate': stats.conversion_rate,
                'filters': {
                    'min_answer_length': self.config.min_answer_length,
                    'min_context_length': self.config.min_context_length,
                    'max_context_length': self.config.max_context_length,
                },
                'statistics': {
                    'avg_context_length': stats.avg_context_length,
                    'avg_answer_length': stats.avg_answer_length,
                    'min_context_length': min(stats.context_lengths) if stats.context_lengths else 0,
                    'max_context_length': max(stats.context_lengths) if stats.context_lengths else 0,
                    'min_answer_length': min(stats.answer_lengths) if stats.answer_lengths else 0,
                    'max_answer_length': max(stats.answer_lengths) if stats.answer_lengths else 0,
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ Results saved successfully")
    
    def _print_summary(self, output_path: str):
        """Print conversion summary."""
        stats = self.converter.stats
        
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Processing Statistics:")
        print(f"  Total examples processed    : {stats.total_processed:,}")
        print(f"  Successfully converted      : {stats.successfully_converted:,}")
        print(f"  Conversion rate             : {stats.conversion_rate:.1%}")
        
        print(f"\n❌ Skipped Examples Breakdown:")
        print(f"  No question text            : {stats.skipped_no_question:,}")
        print(f"  No document tokens          : {stats.skipped_no_document_tokens:,}")
        print(f"  No annotations              : {stats.skipped_no_annotations:,}")
        print(f"  No valid answer             : {stats.skipped_no_valid_answer:,}")
        print(f"  Context too short           : {stats.skipped_context_too_short:,}")
        print(f"  Context too long            : {stats.skipped_context_too_long:,}")
        print(f"  Answer too short            : {stats.skipped_answer_too_short:,}")
        print(f"  Answer not in context       : {stats.skipped_answer_not_in_context:,}")
        print(f"  {'─'*30}")
        print(f"  Total skipped               : {stats.total_skipped:,}")
        
        if stats.context_lengths:
            print(f"\n📏 Context Statistics:")
            print(f"  Average length              : {stats.avg_context_length:,.0f} chars")
            print(f"  Range                       : {min(stats.context_lengths):,} - {max(stats.context_lengths):,} chars")
        
        if stats.answer_lengths:
            print(f"\n📝 Answer Statistics:")
            print(f"  Average length              : {stats.avg_answer_length:.1f} chars")
            print(f"  Range                       : {min(stats.answer_lengths):,} - {max(stats.answer_lengths):,} chars")
        
        print(f"\n📁 Output:")
        print(f"  File: {Path(output_path).absolute()}")
        
        if Path(output_path).exists():
            file_size_kb = Path(output_path).stat().st_size / 1024
            if file_size_kb > 1024:
                print(f"  Size: {file_size_kb / 1024:.1f} MB")
            else:
                print(f"  Size: {file_size_kb:.1f} KB")
        
        print("\n✅ Conversion complete! Ready for evaluation.\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Natural Questions to SQuAD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert first 100 examples
  python nq_to_squad_converter.py --sample 100
  
  # Convert all examples with custom output
  python nq_to_squad_converter.py --output my_data.json
  
  # Enable debug logging
  python nq_to_squad_converter.py --log-level DEBUG --sample 10
        """
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Process only first N examples (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/natural_questions_squad.json)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--min-answer-length',
        type=int,
        default=4,
        help='Minimum answer length in characters (default: 4)'
    )
    
    parser.add_argument(
        '--min-context-length',
        type=int,
        default=50,
        help='Minimum context length in characters (default: 50)'
    )
    
    parser.add_argument(
        '--max-context-length',
        type=int,
        default=50000,
        help='Maximum context length in characters (default: 50000)'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config(
        log_level=args.log_level,
        min_answer_length=args.min_answer_length,
        min_context_length=args.min_context_length,
        max_context_length=args.max_context_length
    )
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    # Create pipeline
    pipeline = ConversionPipeline(config, logger)
    
    # Run conversion
    try:
        pipeline.run(
            sample_size=args.sample,
            output_path=args.output
        )
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n\nConversion interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\nConversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()