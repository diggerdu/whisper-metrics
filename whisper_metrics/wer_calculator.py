"""
Word Error Rate (WER) calculator based on jiwer
"""

import re
from typing import List, Union, Optional, Callable
from jiwer import wer, mer, wil, wip, cer
from jiwer.transforms import (
    AbstractTransform,
    Compose,
    ExpandCommonEnglishContractions,
    ReduceToListOfListOfWords,
    ReduceToListOfListOfChars,
    RemoveKaldiNonWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    RemoveWhiteSpace,
    Strip,
    SubstituteRegexes,
    SubstituteWords,
    ToLowerCase,
)

class WERCalculator:
    """Calculate Word Error Rate and other speech recognition metrics"""
    
    def __init__(self, 
                 word_standardize: bool = True,
                 char_standardize: bool = True,
                 remove_punctuation: bool = True,
                 expand_contractions: bool = True):
        """
        Initialize WER calculator with preprocessing options
        
        Args:
            word_standardize: Apply word-level standardization
            char_standardize: Apply character-level standardization
            remove_punctuation: Remove punctuation marks
            expand_contractions: Expand contractions (e.g., "don't" -> "do not")
        """
        self.word_standardize = word_standardize
        self.char_standardize = char_standardize
        self.remove_punctuation = remove_punctuation
        self.expand_contractions = expand_contractions
        
        self.word_transforms = Compose(self._create_word_transforms())
        self.char_transforms = Compose(self._create_char_transforms())
    
    def _create_word_transforms(self) -> List[AbstractTransform]:
        """Create word-level preprocessing transforms"""
        transforms = []
        
        if self.expand_contractions:
            transforms.append(ExpandCommonEnglishContractions())
        
        if self.word_standardize:
            transforms.extend([
                ToLowerCase(),
                RemoveKaldiNonWords(),
                RemoveWhiteSpace(),
            ])
        
        if self.remove_punctuation:
            transforms.append(RemovePunctuation())
        
        transforms.extend([
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToListOfListOfWords(),
        ])
        
        return transforms
    
    def _create_char_transforms(self) -> List[AbstractTransform]:
        """Create character-level preprocessing transforms"""
        transforms = []
        
        if self.char_standardize:
            transforms.extend([
                ToLowerCase(),
                RemoveWhiteSpace(),
            ])
        
        if self.remove_punctuation:
            transforms.append(RemovePunctuation())
        
        transforms.extend([
            Strip(),
            ReduceToListOfListOfChars(),
        ])
        
        return transforms
    
    def preprocess_text(self, text: str, for_characters: bool = False) -> str:
        """
        Preprocess text for WER calculation
        
        Args:
            text: Input text
            for_characters: Whether preprocessing is for character-level metrics
            
        Returns:
            Preprocessed text
        """
        transforms = self.char_transforms if for_characters else self.word_transforms
        
        processed = text
        for transform in transforms:
            processed = transform(processed)
        
        return processed
    
    def calculate_wer(self, 
                     reference: Union[str, List[str]], 
                     hypothesis: Union[str, List[str]]) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            WER score (0.0 = perfect, higher = worse)
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        
        return wer(reference, hypothesis, 
                  reference_transform=self.word_transforms,
                  hypothesis_transform=self.word_transforms)
    
    def calculate_mer(self, 
                     reference: Union[str, List[str]], 
                     hypothesis: Union[str, List[str]]) -> float:
        """
        Calculate Match Error Rate
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            MER score
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        
        return mer(reference, hypothesis,
                  reference_transform=self.word_transforms,
                  hypothesis_transform=self.word_transforms)
    
    def calculate_wil(self, 
                     reference: Union[str, List[str]], 
                     hypothesis: Union[str, List[str]]) -> float:
        """
        Calculate Word Information Lost
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            WIL score
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        
        return wil(reference, hypothesis,
                  reference_transform=self.word_transforms,
                  hypothesis_transform=self.word_transforms)
    
    def calculate_wip(self, 
                     reference: Union[str, List[str]], 
                     hypothesis: Union[str, List[str]]) -> float:
        """
        Calculate Word Information Preserved
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            WIP score
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        
        return wip(reference, hypothesis,
                  reference_transform=self.word_transforms,
                  hypothesis_transform=self.word_transforms)
    
    def calculate_cer(self, 
                     reference: Union[str, List[str]], 
                     hypothesis: Union[str, List[str]]) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            CER score
        """
        if isinstance(reference, str):
            reference = [reference]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        
        return cer(reference, hypothesis,
                  reference_transform=self.char_transforms,
                  hypothesis_transform=self.char_transforms)
    
    def calculate_all_metrics(self, 
                            reference: Union[str, List[str]], 
                            hypothesis: Union[str, List[str]]) -> dict:
        """
        Calculate all available metrics
        
        Args:
            reference: Reference text(s)
            hypothesis: Hypothesis text(s)
            
        Returns:
            Dictionary with all metric scores
        """
        return {
            'wer': self.calculate_wer(reference, hypothesis),
            'mer': self.calculate_mer(reference, hypothesis),
            'wil': self.calculate_wil(reference, hypothesis),
            'wip': self.calculate_wip(reference, hypothesis),
            'cer': self.calculate_cer(reference, hypothesis),
        }
    
    @staticmethod
    def quick_wer(reference: str, hypothesis: str) -> float:
        """
        Quick WER calculation with default settings
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            WER score
        """
        calculator = WERCalculator()
        return calculator.calculate_wer(reference, hypothesis)