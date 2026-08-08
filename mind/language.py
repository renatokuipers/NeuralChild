"""Language neural network implementation.

This network processes and generates language, evolving from basic vocalizations
to complex sentences as development progresses.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from pydantic import BaseModel, Field, validator
from torch import nn

from core.neural_network import NeuralNetwork
from core.schemas import DevelopmentalStage, NetworkMessage, TextOutput, VectorOutput
from mind.schemas import LanguageAbility

# Configure logging
logger = logging.getLogger(__name__)

class VocabularyEntry(BaseModel):
    """A word in the language network's vocabulary."""

    word: str = Field(..., description="The word itself")
    acquisition_time: datetime = Field(default_factory=datetime.now, description="When this word was learned")
    usage_count: int = Field(default=0, description="How many times this word has been used")
    familiarity: float = Field(default=0.5, ge=0.0, le=1.0, description="How familiar the mind is with this word")
    embedding: Optional[List[float]] = Field(default=None, description="Vector representation of the word")
    associations: Set[str] = Field(default_factory=set, description="Other words associated with this one")
    part_of_speech: Optional[str] = Field(default=None, description="Noun, verb, adjective, etc.")

    class Config:
        arbitrary_types_allowed = True

    @validator("associations", pre=True)
    def ensure_set(cls, v):
        """Ensure associations is a set."""
        if isinstance(v, list):
            return set(v)
        return v

    def increase_familiarity(self, amount: float = 0.05) -> None:
        """Increase familiarity with this word."""
        self.familiarity = min(1.0, self.familiarity + amount)
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "word": self.word,
            "acquisition_time": self.acquisition_time.isoformat(),
            "usage_count": self.usage_count,
            "familiarity": self.familiarity,
            "embedding": self.embedding,
            "associations": list(self.associations),
            "part_of_speech": self.part_of_speech,
        }

class SyntaxRule(BaseModel):
    """A grammatical rule for sentence construction."""

    rule_id: str = Field(..., description="Unique identifier for this rule")
    pattern: List[str] = Field(..., description="Parts of speech pattern, e.g. ['noun', 'verb', 'noun']")
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="How complex this rule is")
    acquisition_stage: DevelopmentalStage = Field(..., description="Stage at which this rule is typically acquired")
    mastery: float = Field(default=0.0, ge=0.0, le=1.0, description="How well this rule has been mastered")
    examples: List[str] = Field(default_factory=list, description="Example sentences using this rule")

    def increase_mastery(self, amount: float = 0.05) -> None:
        """Increase mastery of this syntax rule."""
        self.mastery = min(1.0, self.mastery + amount)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "pattern": self.pattern,
            "complexity": self.complexity,
            "acquisition_stage": self.acquisition_stage.name,
            "mastery": self.mastery,
            "examples": self.examples,
        }

class LanguageNetwork(NeuralNetwork):
    """Language network that processes and generates language.
    
    This network handles language acquisition, evolving from pre-linguistic sounds
    to complex sentences as the mind develops through different stages.
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 192, output_dim: int = 48):
        """Initialize the language network.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors

        """
        super().__init__(name="language", input_dim=input_dim, output_dim=output_dim)

        # Word embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Language generation LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Grammar processing network
        self.grammar_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Output layer for word prediction
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Hidden state for maintaining context
        self.hidden = None
        self.cell = None

        # Vocabulary growth properties
        self.vocabulary: Dict[str, VocabularyEntry] = {}
        self.vocabulary_embedding_dim = 16  # Size of word embeddings

        # Syntax rules by developmental stage
        self.syntax_rules: Dict[str, SyntaxRule] = {}

        # Language abilities that develop over time
        self.language_ability = LanguageAbility(
            vocabulary_size=0,
            sentence_complexity=0.0,
            understanding_level=0.1,
            expression_level=0.0,
        )

        # Recent utterances (both input and output)
        self.recent_utterances: List[Dict[str, Any]] = []

        # Word categories for developmental language learning
        self.word_categories = {
            "nouns": ["mama", "dada", "baby", "toy", "food", "milk", "dog", "cat", "ball"],
            "verbs": ["want", "see", "go", "eat", "play", "sleep", "give", "take"],
            "adjectives": ["big", "small", "good", "bad", "hot", "cold", "happy", "sad"],
            "function_words": ["in", "on", "up", "down", "this", "that", "my", "your", "the", "a"],
        }

        # Initialize basic vocabulary with infant-appropriate words
        self._initialize_vocabulary()

        # Initialize syntax rules
        self._initialize_syntax_rules()

        # Initialize state parameters
        self.update_state({
            "vocabulary_size": len(self.vocabulary),
            "sentence_complexity": self.language_ability.sentence_complexity,
            "understanding_level": self.language_ability.understanding_level,
            "expression_level": self.language_ability.expression_level,
            "recent_utterances": [],
        })

    def _initialize_vocabulary(self) -> None:
        """Initialize basic vocabulary appropriate for infant stage."""
        # Start with a small set of words an infant might first learn
        initial_words = ["mama", "dada", "no", "yes", "milk"]

        for word in initial_words:
            # Create simple random embedding
            embedding = list(np.random.uniform(-1, 1, self.vocabulary_embedding_dim))

            # Determine likely part of speech
            part_of_speech = None
            for category, words in self.word_categories.items():
                if word in words:
                    if category == "nouns":
                        part_of_speech = "noun"
                    elif category == "verbs":
                        part_of_speech = "verb"
                    elif category == "adjectives":
                        part_of_speech = "adjective"
                    elif category == "function_words":
                        part_of_speech = "function"
                    break

            # Add to vocabulary
            self.vocabulary[word] = VocabularyEntry(
                word=word,
                embedding=embedding,
                familiarity=0.3,  # Start with low familiarity
                part_of_speech=part_of_speech,
            )

    def _initialize_syntax_rules(self) -> None:
        """Initialize syntax rules for different developmental stages."""
        # Infant stage - single word utterances
        self.syntax_rules["single_word"] = SyntaxRule(
            rule_id="single_word",
            pattern=["any"],
            complexity=0.1,
            acquisition_stage=DevelopmentalStage.INFANT,
            mastery=0.8,  # Infants are good at single words
            examples=["mama", "dada", "milk"],
        )

        # Toddler stage - two-word combinations
        self.syntax_rules["noun_verb"] = SyntaxRule(
            rule_id="noun_verb",
            pattern=["noun", "verb"],
            complexity=0.3,
            acquisition_stage=DevelopmentalStage.TODDLER,
            mastery=0.0,  # Not yet mastered
            examples=["baby eat", "daddy go"],
        )

        self.syntax_rules["verb_noun"] = SyntaxRule(
            rule_id="verb_noun",
            pattern=["verb", "noun"],
            complexity=0.3,
            acquisition_stage=DevelopmentalStage.TODDLER,
            mastery=0.0,
            examples=["want milk", "see ball"],
        )

        # Child stage - simple sentences
        self.syntax_rules["subject_verb_object"] = SyntaxRule(
            rule_id="subject_verb_object",
            pattern=["noun", "verb", "noun"],
            complexity=0.5,
            acquisition_stage=DevelopmentalStage.CHILD,
            mastery=0.0,
            examples=["I want toy", "Dog eat food"],
        )

        self.syntax_rules["subject_verb_adjective"] = SyntaxRule(
            rule_id="subject_verb_adjective",
            pattern=["noun", "verb", "adjective"],
            complexity=0.5,
            acquisition_stage=DevelopmentalStage.CHILD,
            mastery=0.0,
            examples=["I am happy", "Ball is red"],
        )

        # Adolescent stage - complex sentences
        self.syntax_rules["complex_sentence"] = SyntaxRule(
            rule_id="complex_sentence",
            pattern=["noun", "verb", "noun", "conjunction", "noun", "verb"],
            complexity=0.8,
            acquisition_stage=DevelopmentalStage.ADOLESCENT,
            mastery=0.0,
            examples=["I like dogs but cats are nice", "She reads books when time allows"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.
        
        Args:
            x: Input tensor representing language input
            
        Returns:
            Output tensor representing language processing result

        """
        batch_size = x.size(0)

        # Process through embedding network
        embedded = self.embedding_network(x)

        # Process embedded input through LSTM for sequential processing
        if x.dim() == 2:
            # Add sequence dimension if not present
            embedded = embedded.unsqueeze(1)

        # Initialize hidden state if None
        if self.hidden is None or self.cell is None:
            self.hidden = torch.zeros(2, batch_size, self.lstm.hidden_size, device=x.device)
            self.cell = torch.zeros(2, batch_size, self.lstm.hidden_size, device=x.device)

        # Process through LSTM
        lstm_out, (self.hidden, self.cell) = self.lstm(embedded, (self.hidden, self.cell))

        # Get last output from LSTM
        last_output = lstm_out[:, -1, :]

        # Apply grammar processing based on developmental stage
        if self.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
            # Apply grammar network for more complex processing
            grammar_results = self.grammar_network(last_output)

            # Blend grammar processing with LSTM output based on developmental stage
            grammar_weight = min(1.0, 0.2 * (self.developmental_stage.value - 1))
            blended = last_output * (1 - grammar_weight) + grammar_weight * torch.cat([grammar_results, torch.zeros(batch_size, self.lstm.hidden_size - self.output_dim, device=x.device)], dim=1)
            result = self.output_layer(blended)
        else:
            # Simple processing for infant stage
            result = self.output_layer(last_output)

        # Scale output based on language ability
        scaled_result = result * self.language_ability.expression_level

        return scaled_result

    def process_message(self, message: NetworkMessage) -> Optional[VectorOutput]:
        """Process a message from another neural network.
        
        Args:
            message: Message from another network
            
        Returns:
            Optional vector output as response

        """
        # Process language input from perception
        if message.message_type == "language_input" and "text" in message.content:
            text_input = message.content["text"]

            # Process and understand the input
            understanding = self._process_language_input(text_input)

            # Create a vector representation
            if "vector_data" in message.content:
                vector_data = message.content["vector_data"]

                # Ensure vector is the right size
                if len(vector_data) > self.input_dim:
                    vector_data = vector_data[:self.input_dim]
                elif len(vector_data) < self.input_dim:
                    vector_data = vector_data + [0.0] * (self.input_dim - len(vector_data))

                # Convert to tensor and process
                input_tensor = torch.tensor(vector_data, dtype=torch.float32)

                with torch.no_grad():
                    output_tensor = self.forward(input_tensor.unsqueeze(0))

                return VectorOutput(
                    source=self.name,
                    data=output_tensor[0].tolist(),
                )

            # If no vector data, create a simple response based on text
            simple_vector = [0.0] * self.output_dim
            simple_vector[0] = 1.0  # Signal that we received input

            return VectorOutput(
                source=self.name,
                data=simple_vector,
            )

        # Generate language response to emotions
        if message.message_type == "emotion" and self.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
            if "emotion" in message.content and "intensity" in message.content:
                emotion = message.content["emotion"]
                intensity = float(message.content["intensity"])

                # Generate a response appropriate to the emotion
                response = self._generate_emotion_response(emotion, intensity)

                # Create a vector to return
                response_vector = [0.0] * self.output_dim
                response_vector[1] = intensity  # Use slot 1 for emotional response intensity

                # Send a message to consciousness with the generated text
                self._send_language_output(response)

                return VectorOutput(
                    source=self.name,
                    data=response_vector,
                )

        # Respond to query about language ability
        elif message.message_type == "query":
            if "query_type" in message.content and message.content["query_type"] == "language_ability":
                # Return vector representation of language ability
                ability_vector = [
                    min(1.0, len(self.vocabulary) / 1000),  # Normalized vocabulary size
                    self.language_ability.sentence_complexity,
                    self.language_ability.understanding_level,
                    self.language_ability.expression_level,
                ]

                # Pad to output dimension
                ability_vector = ability_vector + [0.0] * (self.output_dim - len(ability_vector))

                return VectorOutput(
                    source=self.name,
                    data=ability_vector,
                )

        return None

    def _process_language_input(self, text: str) -> Dict[str, Any]:
        """Process language input and update vocabulary.
        
        Args:
            text: Text input to process
            
        Returns:
            Dictionary with processing results

        """
        # Simple preprocessing - lowercase and strip
        text = text.lower().strip()

        # Check if the input is a pre-linguistic sound (non-word)
        if not any(c.isalpha() for c in text):
            # Just add to recent utterances and return basic understanding
            self.recent_utterances.append({
                "type": "input",
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "words_recognized": 0,
                "understanding_level": 0.1,
            })

            return {
                "type": "pre_linguistic",
                "understanding_level": 0.1,
                "words_recognized": 0,
            }

        # Split into words for more developed processing
        words = text.split()
        words_recognized = 0
        new_words = []

        # Process each word
        for word in words:
            # Clean the word of punctuation for matching
            clean_word = "".join(c for c in word if c.isalpha())
            if not clean_word:
                continue

            if clean_word in self.vocabulary:
                # Increase familiarity with known word
                self.vocabulary[clean_word].increase_familiarity()
                words_recognized += 1

                # Update associations between adjacent words
                if len(new_words) > 0 and new_words[-1] in self.vocabulary:
                    prev_word = new_words[-1]
                    self.vocabulary[clean_word].associations.add(prev_word)
                    self.vocabulary[prev_word].associations.add(clean_word)
            else:
                # Learn new word if mind is capable based on developmental stage
                learn_probability = 0.1 * self.developmental_stage.value * self.language_ability.understanding_level
                if random.random() < learn_probability:
                    # Create embedding for new word
                    embedding = list(np.random.uniform(-1, 1, self.vocabulary_embedding_dim))

                    # Guess part of speech based on position and patterns
                    part_of_speech = self._guess_part_of_speech(clean_word, words, words.index(word))

                    # Add to vocabulary
                    self.vocabulary[clean_word] = VocabularyEntry(
                        word=clean_word,
                        embedding=embedding,
                        familiarity=0.1,  # Start with low familiarity
                        part_of_speech=part_of_speech,
                    )

                    new_words.append(clean_word)
                    logger.info(f"Learned new word: {clean_word} (guessed POS: {part_of_speech})")

        # Update language ability based on vocabulary size
        self.language_ability.vocabulary_size = len(self.vocabulary)

        # Calculate understanding level based on recognized words
        if len(words) > 0:
            understanding = min(1.0, words_recognized / len(words)) * self.language_ability.understanding_level
        else:
            understanding = 0.1

        # Add to recent utterances
        self.recent_utterances.append({
            "type": "input",
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "words_recognized": words_recognized,
            "understanding_level": understanding,
        })

        # Limit recent utterances size
        if len(self.recent_utterances) > 10:
            self.recent_utterances = self.recent_utterances[-10:]

        # Update state
        self.update_state({
            "vocabulary_size": len(self.vocabulary),
            "recent_utterances": self.recent_utterances,
            "understanding_level": self.language_ability.understanding_level,
        })

        return {
            "type": "words",
            "words_recognized": words_recognized,
            "total_words": len(words),
            "understanding_level": understanding,
            "new_words": new_words,
        }

    def _guess_part_of_speech(self, word: str, sentence: List[str], position: int) -> Optional[str]:
        """Guess the part of speech of a word based on context.
        
        Args:
            word: Word to analyze
            sentence: Complete sentence
            position: Position of word in sentence
            
        Returns:
            Guessed part of speech or None

        """
        # Check if word is in our categories
        for category, words in self.word_categories.items():
            if word in words:
                if category == "nouns":
                    return "noun"
                if category == "verbs":
                    return "verb"
                if category == "adjectives":
                    return "adjective"
                if category == "function_words":
                    return "function"

        # Use position-based heuristics
        if position == 0:
            # First word often a noun or pronoun
            return "noun"

        if position == 1 and len(sentence) >= 3:
            # Second word in 3+ word sentence often a verb
            return "verb"

        # Check surrounding known words
        surrounding_words = []
        if position > 0 and sentence[position-1] in self.vocabulary:
            surrounding_words.append(self.vocabulary[sentence[position-1]])

        if position < len(sentence)-1 and sentence[position+1] in self.vocabulary:
            surrounding_words.append(self.vocabulary[sentence[position+1]])

        # If we find patterns like adjective + unknown, the unknown is likely a noun
        if any(w.part_of_speech == "adjective" for w in surrounding_words):
            return "noun"

        # Return None if we can't determine
        return None

    def _generate_emotion_response(self, emotion: str, intensity: float) -> str:
        """Generate language response to an emotion.
        
        Args:
            emotion: Emotion name
            intensity: Emotion intensity
            
        Returns:
            Generated text response

        """
        # Response based on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            # Infants make simple sounds
            if emotion in ["joy", "trust"]:
                return random.choice(["goo", "gah", "aah"])
            if emotion in ["sadness", "fear"]:
                return random.choice(["waa", "uuh"])
            return random.choice(["ah", "oh", "eh"])

        if self.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers use simple words
            if emotion == "joy":
                return random.choice(["happy", "good", "yay", "like"])
            if emotion == "sadness":
                return random.choice(["sad", "no", "bad"])
            if emotion == "fear":
                return random.choice(["scared", "no", "bad"])
            if emotion == "anger":
                return random.choice(["mad", "no", "bad"])
            # Use a random vocabulary word
            if self.vocabulary:
                return random.choice(list(self.vocabulary.keys()))
            return "oh"

        if self.developmental_stage == DevelopmentalStage.CHILD:
            # Children use simple sentences
            if emotion == "joy":
                return random.choice([
                    "I am happy",
                    "I like this",
                    "This is good",
                    "I feel good",
                ])
            if emotion == "sadness":
                return random.choice([
                    "I am sad",
                    "I feel bad",
                    "This makes me sad",
                    "I don't like this",
                ])
            if emotion == "fear":
                return random.choice([
                    "I am scared",
                    "This is scary",
                    "I feel afraid",
                    "I don't like this",
                ])
            if emotion == "anger":
                return random.choice([
                    "I am mad",
                    "This makes me angry",
                    "I don't like this",
                    "Stop this",
                ])
            return self._generate_simple_sentence()

        # More complex sentences
        if emotion == "joy":
            return random.choice([
                "I'm feeling really happy about this.",
                "This brings me joy.",
                "I'm experiencing positive emotions right now.",
                "This makes me feel good.",
            ])
        if emotion == "sadness":
            return random.choice([
                "I'm feeling sad about this situation.",
                "This is making me feel down.",
                "I'm experiencing sadness right now.",
                "This is disappointing to me.",
            ])
        if emotion == "fear":
            return random.choice([
                "This situation makes me feel anxious.",
                "I'm experiencing fear right now.",
                "I'm concerned about what's happening.",
                "This is making me feel uncomfortable.",
            ])
        if emotion == "anger":
            return random.choice([
                "I'm feeling frustrated about this.",
                "This situation is making me angry.",
                "I'm experiencing irritation right now.",
                "This is really bothering me.",
            ])
        return self._generate_complex_sentence()

    def _generate_simple_sentence(self) -> str:
        """Generate a simple sentence based on vocabulary and syntax rules.
        
        Returns:
            Generated sentence

        """
        # Get appropriate rules for current stage
        available_rules = [
            rule for rule in self.syntax_rules.values()
            if rule.acquisition_stage.value <= self.developmental_stage.value
            and rule.mastery > 0.3
        ]

        if not available_rules:
            # Fallback to single words if no rules are mastered enough
            if self.vocabulary:
                return random.choice(list(self.vocabulary.keys()))
            return ""

        # Select a rule weighted by mastery
        total_mastery = sum(rule.mastery for rule in available_rules)
        r = random.random() * total_mastery

        selected_rule = available_rules[0]
        cumulative = 0
        for rule in available_rules:
            cumulative += rule.mastery
            if r <= cumulative:
                selected_rule = rule
                break

        # Follow the pattern to build a sentence
        sentence_parts = []

        for part in selected_rule.pattern:
            if part == "any":
                # Any word will do
                if self.vocabulary:
                    sentence_parts.append(random.choice(list(self.vocabulary.keys())))
            else:
                # Find words matching the part of speech
                matching_words = [
                    word for word, entry in self.vocabulary.items()
                    if entry.part_of_speech == part and entry.familiarity > 0.3
                ]

                if matching_words:
                    sentence_parts.append(random.choice(matching_words))
                elif part == "noun":
                    sentence_parts.append(random.choice(self.word_categories["nouns"]))
                elif part == "verb":
                    sentence_parts.append(random.choice(self.word_categories["verbs"]))
                elif part == "adjective":
                    sentence_parts.append(random.choice(self.word_categories["adjectives"]))
                elif part == "function":
                    sentence_parts.append(random.choice(self.word_categories["function_words"]))
                # Unknown part of speech, use anything
                elif self.vocabulary:
                    sentence_parts.append(random.choice(list(self.vocabulary.keys())))

        # Increase mastery of this rule slightly
        selected_rule.increase_mastery(0.01)

        # Join parts into a sentence
        return " ".join(sentence_parts)

    def _generate_complex_sentence(self) -> str:
        """Generate a more complex sentence for advanced stages.
        
        Returns:
            Generated complex sentence

        """
        # For simplicity, generate two simple sentences and join them
        conjunction = random.choice(["and", "but", "because", "when"])

        first_part = self._generate_simple_sentence()
        second_part = self._generate_simple_sentence()

        return f"{first_part} {conjunction} {second_part}"

    def _send_language_output(self, text: str) -> None:
        """Send generated language to other networks.
        
        Args:
            text: Generated text

        """
        # Add to recent utterances
        self.recent_utterances.append({
            "type": "output",
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })

        # Limit recent utterances size
        if len(self.recent_utterances) > 10:
            self.recent_utterances = self.recent_utterances[-10:]

        # Update state
        self.update_state({
            "recent_utterances": self.recent_utterances,
        })

        # Create message to send to consciousness
        language_message = NetworkMessage(
            sender=self.name,
            receiver="consciousness",
            message_type="language_output",
            content={
                "text": text,
                "complexity": self.language_ability.sentence_complexity,
                "expression_level": self.language_ability.expression_level,
            },
            priority=0.7,
        )

        # Add to state for the mind to retrieve
        self.update_state({
            "pending_messages": self.state.parameters.get("pending_messages", []) + [language_message.to_dict()],
        })

    def autonomous_step(self) -> None:
        """Autonomous processing step.
        
        This function is called periodically by the mind to allow
        the network to perform autonomous processing.
        """
        # Spontaneous utterance probability based on developmental stage
        spontaneous_utterance_prob = 0.05 * self.developmental_stage.value

        if random.random() < spontaneous_utterance_prob:
            # Generate a spontaneous utterance
            utterance = self._generate_spontaneous_utterance()

            if utterance:
                self._send_language_output(utterance)

        # Consolidate word associations
        self._consolidate_word_associations()

    def _generate_spontaneous_utterance(self) -> Optional[str]:
        """Generate a spontaneous utterance based on current state.
        
        Returns:
            Generated utterance or None

        """
        # No utterances at very low expression levels
        if self.language_ability.expression_level < 0.1:
            return None

        # Generate based on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            # Infants make simple sounds
            sounds = ["goo", "gah", "bah", "mah", "dah", "ah", "oh"]
            return random.choice(sounds)

        if self.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers use single words or two-word combinations
            if random.random() < 0.7 or len(self.vocabulary) < 5:
                # Single word
                if self.vocabulary:
                    return random.choice(list(self.vocabulary.keys()))
                return random.choice(self.word_categories["nouns"])
            # Two-word combination
            if len(self.vocabulary) >= 2:
                words = random.sample(list(self.vocabulary.keys()), 2)
                return f"{words[0]} {words[1]}"
            return random.choice(self.word_categories["nouns"])

        if self.developmental_stage == DevelopmentalStage.CHILD:
            # Children use simple sentences
            return self._generate_simple_sentence()

        # More complex sentences
        if random.random() < 0.7:
            return self._generate_simple_sentence()
        return self._generate_complex_sentence()

    def _consolidate_word_associations(self) -> None:
        """Consolidate and strengthen word associations based on usage patterns."""
        # Skip if vocabulary is too small
        if len(self.vocabulary) < 5:
            return

        # Select a few random words to process
        sample_size = min(5, len(self.vocabulary))
        sample_words = random.sample(list(self.vocabulary.keys()), sample_size)

        for word in sample_words:
            word_entry = self.vocabulary[word]

            # Strengthen connections between frequently used words
            if word_entry.associations:
                associated_words = list(word_entry.associations)

                # Reinforce a random association
                if associated_words:
                    assoc_word = random.choice(associated_words)
                    if assoc_word in self.vocabulary:
                        # Mutually reinforce the association
                        self.vocabulary[assoc_word].associations.add(word)

                        # Increase familiarity slightly
                        word_entry.increase_familiarity(0.01)
                        self.vocabulary[assoc_word].increase_familiarity(0.01)

    def update_developmental_stage(self, stage: DevelopmentalStage) -> None:
        """Update the developmental stage of the network.
        
        As the network develops, language abilities improve and more complex
        linguistic structures become available.
        
        Args:
            stage: New developmental stage

        """
        super().update_developmental_stage(stage)

        # Update language abilities based on developmental stage
        stage_values = {
            DevelopmentalStage.INFANT: {
                "sentence_complexity": 0.0,
                "understanding_level": 0.1,
                "expression_level": 0.1,
            },
            DevelopmentalStage.TODDLER: {
                "sentence_complexity": 0.2,
                "understanding_level": 0.3,
                "expression_level": 0.3,
            },
            DevelopmentalStage.CHILD: {
                "sentence_complexity": 0.5,
                "understanding_level": 0.6,
                "expression_level": 0.6,
            },
            DevelopmentalStage.ADOLESCENT: {
                "sentence_complexity": 0.8,
                "understanding_level": 0.9,
                "expression_level": 0.8,
            },
            DevelopmentalStage.MATURE: {
                "sentence_complexity": 1.0,
                "understanding_level": 1.0,
                "expression_level": 1.0,
            },
        }

        if stage in stage_values:
            ability_values = stage_values[stage]
            self.language_ability.sentence_complexity = ability_values["sentence_complexity"]
            self.language_ability.understanding_level = ability_values["understanding_level"]
            self.language_ability.expression_level = ability_values["expression_level"]

            # Update state
            self.update_state({
                "sentence_complexity": self.language_ability.sentence_complexity,
                "understanding_level": self.language_ability.understanding_level,
                "expression_level": self.language_ability.expression_level,
            })

        # Add stage-appropriate vocabulary if advancing to a new stage
        self._expand_vocabulary_for_stage(stage)

        # Update mastery of syntax rules based on stage
        for rule in self.syntax_rules.values():
            if rule.acquisition_stage.value < stage.value:
                # Increase mastery of rules from previous stages
                rule.increase_mastery(0.2)
            elif rule.acquisition_stage == stage:
                # Initialize mastery for current stage rules
                rule.increase_mastery(0.1)

        # Reset hidden state to accommodate new stage capabilities
        self.hidden = None
        self.cell = None

        logger.info(f"Language network updated to {stage.name} stage")

    def _expand_vocabulary_for_stage(self, stage: DevelopmentalStage) -> None:
        """Expand vocabulary with words appropriate for the developmental stage.
        
        Args:
            stage: Developmental stage to expand vocabulary for

        """
        # Words to potentially add for each stage
        stage_vocabulary = {
            DevelopmentalStage.INFANT: ["mama", "dada", "no", "hi", "baba", "milk"],
            DevelopmentalStage.TODDLER: ["me", "you", "want", "more", "eat", "play", "big", "hot", "cold", "go", "mine"],
            DevelopmentalStage.CHILD: ["I", "am", "the", "a", "and", "but", "like", "don't", "because", "what", "who", "when", "where", "good", "bad", "happy", "sad"],
            DevelopmentalStage.ADOLESCENT: ["think", "feel", "believe", "understand", "sometimes", "never", "always", "maybe", "probably", "interesting", "boring"],
            DevelopmentalStage.MATURE: ["however", "therefore", "although", "nevertheless", "furthermore", "consider", "hypothetical", "perspective"],
        }

        # Add words for current and all previous stages
        for s in DevelopmentalStage:
            if s.value <= stage.value:
                words_to_add = stage_vocabulary.get(s, [])
                for word in words_to_add:
                    if word not in self.vocabulary:
                        # Create embedding
                        embedding = list(np.random.uniform(-1, 1, self.vocabulary_embedding_dim))

                        # Determine part of speech
                        part_of_speech = None
                        for category, words in self.word_categories.items():
                            if word in words:
                                if category == "nouns":
                                    part_of_speech = "noun"
                                elif category == "verbs":
                                    part_of_speech = "verb"
                                elif category == "adjectives":
                                    part_of_speech = "adjective"
                                elif category == "function_words":
                                    part_of_speech = "function"
                                break

                        # Add to vocabulary with appropriate familiarity for stage
                        familiarity = 0.3 + (0.1 * (stage.value - s.value))
                        self.vocabulary[word] = VocabularyEntry(
                            word=word,
                            embedding=embedding,
                            familiarity=min(0.7, familiarity),
                            part_of_speech=part_of_speech,
                        )

        # Update vocabulary size
        self.language_ability.vocabulary_size = len(self.vocabulary)
        self.update_state({"vocabulary_size": len(self.vocabulary)})

    def generate_text_output(self) -> TextOutput:
        """Generate a human-readable text output from the neural network.
        
        Returns:
            Text representation of the network's current state

        """
        # Generate text based on language ability and recent utterances
        if not self.recent_utterances:
            if self.developmental_stage == DevelopmentalStage.INFANT:
                text = "Pre-linguistic sounds and coos."
            else:
                text = "No recent language activity."

            return TextOutput(
                source=self.name,
                text=text,
                confidence=0.5,
            )

        # Get most recent utterance
        latest = self.recent_utterances[-1]

        # Base text on developmental stage
        if self.developmental_stage == DevelopmentalStage.INFANT:
            text = f"Vocalizing: {latest.get('text', '')}"
            confidence = 0.3

        elif self.developmental_stage == DevelopmentalStage.TODDLER:
            text = f"Expressing: {latest.get('text', '')}"
            text += f" (Vocabulary: {len(self.vocabulary)} words)"
            confidence = 0.5

        elif self.developmental_stage == DevelopmentalStage.CHILD:
            text = f"Saying: {latest.get('text', '')}"
            text += f" | Vocabulary: {len(self.vocabulary)} words, "
            text += f"Sentence complexity: {int(self.language_ability.sentence_complexity * 100)}%"
            confidence = 0.7

        else:  # ADOLESCENT or MATURE
            text = f"Communication: {latest.get('text', '')}"
            text += f" | Language development: Understanding {int(self.language_ability.understanding_level * 100)}%, "
            text += f"Expression {int(self.language_ability.expression_level * 100)}%"
            confidence = 0.8

        return TextOutput(
            source=self.name,
            text=text,
            confidence=confidence,
        )

    def learn_from_interaction(self, input_text: str, response_text: str) -> None:
        """Learn from an interaction between input and output text.
        
        Args:
            input_text: Input text received
            response_text: Response text generated

        """
        # Process input and response
        input_understanding = self._process_language_input(input_text)

        # Extract patterns from interaction
        if self.developmental_stage.value >= DevelopmentalStage.TODDLER.value:
            input_words = input_text.lower().strip().split()
            response_words = response_text.lower().strip().split()

            # Look for recurring patterns
            if len(input_words) >= 2 and len(response_words) >= 2:
                # Very simple pattern recognition - look for word order patterns
                for i in range(len(input_words) - 1):
                    word_pair = (input_words[i], input_words[i+1])

                    # Check if this pair appears in response
                    for j in range(len(response_words) - 1):
                        if response_words[j] == word_pair[0] and response_words[j+1] == word_pair[1]:
                            # Pattern found - reinforce it
                            for word in word_pair:
                                if word in self.vocabulary:
                                    self.vocabulary[word].increase_familiarity(0.05)

                            # If words have part of speech, try to identify a syntactic pattern
                            if all(word in self.vocabulary for word in word_pair):
                                pos = [self.vocabulary[word].part_of_speech for word in word_pair]
                                if all(pos):
                                    pattern_key = f"{pos[0]}_{pos[1]}"

                                    # Check if we have a rule for this pattern
                                    found_rule = False
                                    for rule in self.syntax_rules.values():
                                        if len(rule.pattern) == 2 and rule.pattern[0] == pos[0] and rule.pattern[1] == pos[1]:
                                            rule.increase_mastery(0.05)
                                            if len(rule.examples) < 5:
                                                example = f"{word_pair[0]} {word_pair[1]}"
                                                if example not in rule.examples:
                                                    rule.examples.append(example)
                                            found_rule = True
                                            break

                                    # Create a new rule if pattern not found and advanced enough
                                    if not found_rule and self.developmental_stage.value >= DevelopmentalStage.CHILD.value:
                                        rule_id = f"learned_{pattern_key}_{len(self.syntax_rules)}"
                                        self.syntax_rules[rule_id] = SyntaxRule(
                                            rule_id=rule_id,
                                            pattern=[pos[0], pos[1]],
                                            complexity=0.4,
                                            acquisition_stage=self.developmental_stage,
                                            mastery=0.1,
                                            examples=[f"{word_pair[0]} {word_pair[1]}"],
                                        )

        # Increase understanding level slightly from this interaction
        understanding_gain = 0.001 * self.developmental_stage.value
        self.language_ability.understanding_level = min(1.0, self.language_ability.understanding_level + understanding_gain)

        # Increase expression level if we generated a response
        if response_text:
            expression_gain = 0.001 * self.developmental_stage.value
            self.language_ability.expression_level = min(1.0, self.language_ability.expression_level + expression_gain)

        # Update state
        self.update_state({
            "understanding_level": self.language_ability.understanding_level,
            "expression_level": self.language_ability.expression_level,
        })

    def clone_with_growth(self, growth_factor: float = 1.2, min_dim: int = 8) -> "LanguageNetwork":
        """Create a larger clone of this network with scaled dimensions.
        
        Args:
            growth_factor: Factor to scale dimensions by
            min_dim: Minimum dimension size to ensure
            
        Returns:
            Larger clone of this network with scaled dimensions

        """
        # Calculate new dimensions
        new_input_dim = max(min_dim, int(self.input_dim * growth_factor))
        new_hidden_dim = max(min_dim * 2, int(self.lstm.hidden_size * growth_factor))
        new_output_dim = max(min_dim, int(self.output_dim * growth_factor))

        # Create new network with expanded dimensions
        new_network = LanguageNetwork(
            input_dim=new_input_dim,
            hidden_dim=new_hidden_dim,
            output_dim=new_output_dim,
        )

        # Transfer language properties
        new_network.vocabulary = copy.deepcopy(self.vocabulary)
        new_network.vocabulary_embedding_dim = int(self.vocabulary_embedding_dim * growth_factor)
        new_network.syntax_rules = copy.deepcopy(self.syntax_rules)
        new_network.language_ability = copy.deepcopy(self.language_ability)
        new_network.recent_utterances = copy.deepcopy(self.recent_utterances)

        # Transfer growth metrics
        new_network.growth_metrics = copy.deepcopy(self.growth_metrics)
        new_network.experience_count = self.experience_count

        # Record growth event
        new_network.growth_history = copy.deepcopy(self.growth_history)
        new_network.growth_history.append(NeuralGrowthRecord(
            event_type="network_expansion",
            layer_affected="all",
            old_shape=[self.input_dim, self.lstm.hidden_size, self.output_dim],
            new_shape=[new_input_dim, new_hidden_dim, new_output_dim],
            growth_factor=growth_factor,
            trigger="clone_with_growth",
            developmental_stage=self.developmental_stage,
        ))

        # Reset hidden states
        new_network.hidden = None
        new_network.cell = None

        logger.info(
            f"LanguageNetwork cloned with growth factor {growth_factor}: "
            f"({self.input_dim}, {self.lstm.hidden_size}, {self.output_dim}) → "
            f"({new_input_dim}, {new_hidden_dim}, {new_output_dim})",
        )

        return new_network
