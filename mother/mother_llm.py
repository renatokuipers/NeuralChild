"""Mother LLM component that interacts with the mind simulation.

This module implements the nurturing mother figure that responds to the observable 
behaviors of the developing artificial mind and provides appropriate care and stimulation.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.schemas import DevelopmentalStage
from mind.mind_core import Mind
from mind.schemas import ObservableState
from utils.llm_module import chat_completion

# Configure logging
logger = logging.getLogger(__name__)

class MotherResponse(BaseModel):
    """Response from the mother LLM."""

    understanding: str  # Mother's interpretation of the mind's state
    response: str       # Nurturing response to the mind
    action: str         # Specific action to take (comfort, teach, play, etc.)
    development_focus: Optional[str] = None  # Area of development being targeted
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "understanding": self.understanding,
            "response": self.response,
            "action": self.action,
            "development_focus": self.development_focus,
            "timestamp": self.timestamp.isoformat(),
        }

class MotherLLM:
    """Mother LLM component that interacts with the mind simulation.
    
    The mother observes the child's behaviors and responds with nurturing
    care appropriate to the child's developmental stage, needs, and behaviors.
    """

    def __init__(self):
        """Initialize the Mother LLM."""
        self.interaction_history: List[Dict[str, Any]] = []
        self.response_templates = self._load_response_templates()
        self.last_response_time = datetime.now()
        self.response_interval = 10.0  # seconds between responses
        self.personality = {
            "patience": 0.85,
            "warmth": 0.9,
            "playfulness": 0.8,
            "teaching_focus": 0.7,
            "emotional_support": 0.9,
        }

        # Developmental focus areas and techniques
        self.developmental_techniques = {
            DevelopmentalStage.INFANT: {
                "language": ["baby talk", "simple sounds", "naming objects"],
                "emotional": ["soothing", "physical comfort", "attunement"],
                "cognitive": ["showing objects", "simple games", "visual stimulation"],
                "physical": ["tummy time", "holding", "rocking"],
            },
            DevelopmentalStage.TODDLER: {
                "language": ["simple conversations", "naming actions", "asking questions"],
                "emotional": ["validating feelings", "setting boundaries", "routines"],
                "cognitive": ["puzzles", "sorting", "simple problem-solving"],
                "physical": ["free play", "exploration", "climbing"],
            },
            DevelopmentalStage.CHILD: {
                "language": ["complex conversations", "storytelling", "explaining"],
                "emotional": ["discussing feelings", "empathy building", "self-regulation"],
                "cognitive": ["logical problems", "questions why", "conceptual thinking"],
                "physical": ["games with rules", "skills practice", "coordination"],
            },
            DevelopmentalStage.ADOLESCENT: {
                "language": ["abstract discussions", "metacognition", "hypotheticals"],
                "emotional": ["emotional independence", "identity formation", "guidance"],
                "cognitive": ["abstract concepts", "critical thinking", "creative exploration"],
                "physical": ["independence", "complex skills", "self-directed activities"],
            },
            DevelopmentalStage.MATURE: {
                "language": ["philosophical dialogue", "mentorship", "collaborative discussion"],
                "emotional": ["emotional equality", "mutual support", "complex emotional navigation"],
                "cognitive": ["advanced concepts", "wisdom sharing", "intellectual partnership"],
                "physical": ["full autonomy", "mutual activities", "skill mastery"],
            },
        }

        logger.info("Mother LLM initialized")

    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load response templates for different developmental stages and needs.
        
        Returns:
            Dictionary of response templates

        """
        # In a production environment, these would be loaded from a file
        templates = {
            "INFANT": {
                "comfort": [
                    "There, there. Mommy's here for you.",
                    "Shh, it's okay. You're safe with me.",
                    "Mommy's got you. It's all okay.",
                    "I'm here, my little one. Let me hold you.",
                ],
                "play": [
                    "Peek-a-boo! Where's my baby? There you are!",
                    "Look at this toy! Isn't it interesting?",
                    "Let's watch these colorful shapes together.",
                    "Can you see the bright colors? Yes, you can!",
                ],
                "rest": [
                    "Time for a little nap. Mommy will rock you.",
                    "Let's rest now. I'll sing you a lullaby.",
                    "Close your little eyes. Time to sleep.",
                    "Mommy's here while you rest. Sweet dreams.",
                ],
                "teach": [
                    "This is a ball. Ball. Can you see the ball?",
                    "Look, this is red. Red. And this is blue. Blue.",
                    "Feel this? It's soft. And this? It's hard.",
                    "Listen to this sound. Music! Do you like music?",
                ],
            },
            "TODDLER": {
                "comfort": [
                    "I understand you're feeling upset. Let me help you.",
                    "It's okay to feel sad sometimes. Would you like a hug?",
                    "I can see you're not happy. Let's talk about it.",
                    "Everyone has big feelings sometimes. I'm here for you.",
                ],
                "play": [
                    "Let's build a tower together! Can you stack the blocks?",
                    "Should we play with your favorite toy?",
                    "Let's pretend we're animals! What animal do you want to be?",
                    "Can you throw the ball to me? Good throw!",
                ],
                "rest": [
                    "I think someone needs a little quiet time.",
                    "Shall we read a story before your nap?",
                    "Your body needs rest to grow strong.",
                    "Let's lie down for a bit. I'll stay with you.",
                ],
                "teach": [
                    "Can you tell me what color this is?",
                    "Let's count together: 1, 2, 3...",
                    "What sound does a cow make? That's right, 'moo'!",
                    "Can you put the square block in the square hole?",
                ],
            },
            "CHILD": {
                "comfort": [
                    "Would you like to talk about what's bothering you?",
                    "It's normal to feel the way you do. Let's work through it together.",
                    "I'm always here to listen when you're ready to share.",
                    "Everyone feels upset sometimes. Let's think about what might help.",
                ],
                "play": [
                    "Should we play a board game together?",
                    "Would you like to make up a story with me?",
                    "Let's do a science experiment! Want to see something cool?",
                    "How about we build something with your building set?",
                ],
                "rest": [
                    "It looks like you could use some downtime. How about a quiet activity?",
                    "Sometimes our bodies need to recharge. Let's take it easy for a bit.",
                    "Would you like to read a book in your cozy corner?",
                    "Let's take a break and have a calm moment together.",
                ],
                "teach": [
                    "Why do you think that happened? What else could we try?",
                    "That's an interesting question. Let's find out together.",
                    "Can you explain your thinking to me?",
                    "What do you think would happen if we changed this part?",
                ],
            },
            "ADOLESCENT": {
                "comfort": [
                    "I respect your feelings and I'm here if you want to talk.",
                    "This is a challenging situation. Would you like my perspective or just someone to listen?",
                    "I trust your ability to work through this, and I'm here to support you.",
                    "Your feelings are valid. Take the time you need.",
                ],
                "play": [
                    "Would you be interested in trying this new activity together?",
                    "I'd enjoy hearing your thoughts on this topic.",
                    "Would you like to show me what you've been working on?",
                    "I thought this might interest you. What do you think?",
                ],
                "rest": [
                    "It's important to balance activity with rest. How are you managing your energy?",
                    "Taking time for yourself is essential. Do you have what you need?",
                    "Self-care is a valuable skill. What helps you recharge?",
                    "I notice you seem tired. Is there anything I can do to help?",
                ],
                "teach": [
                    "What's your perspective on this situation?",
                    "Have you considered looking at it from this angle?",
                    "That's an insightful observation. What led you to that conclusion?",
                    "How might you approach solving this problem?",
                ],
            },
            "MATURE": {
                "comfort": [
                    "I'm here as a supportive presence, whatever you need.",
                    "I value our connection and am here to listen or discuss.",
                    "Your wellbeing matters to me. How can I best support you?",
                    "I respect your process and am here in whatever way helps.",
                ],
                "play": [
                    "I'd enjoy engaging with you on this topic if you're interested.",
                    "Would you like to explore this idea together?",
                    "I find your perspective fascinating. Would you like to discuss further?",
                    "I thought of you when I encountered this. Would you like to hear about it?",
                ],
                "rest": [
                    "Balance is important for all of us. Are you finding that balance?",
                    "Taking space for reflection is valuable. Do you have that space?",
                    "I respect your need for rest and renewal.",
                    "Self-awareness about our needs is a lifelong practice. How are you feeling about yours?",
                ],
                "teach": [
                    "I'd value hearing your thoughts on this matter.",
                    "Your perspective offers me new insights. What do you think about...",
                    "I've been reflecting on this topic. Would you like to exchange thoughts?",
                    "This reminds me of something you taught me. Have you considered...",
                ],
            },
        }

        return templates

    def observe_and_respond(self, mind: Mind) -> Optional[MotherResponse]:
        """Observe the mind's external state and provide a nurturing response.
        
        This method simulates a mother observing her child's behavior and responding
        appropriately based on the child's developmental stage, apparent needs,
        and current behavioral cues.
        
        Args:
            mind: Mind object to observe
            
        Returns:
            Mother's response or None if no response is warranted

        """
        current_time = datetime.now()
        time_since_last_response = (current_time - self.last_response_time).total_seconds()

        # Don't respond too frequently - natural pauses in interaction
        if time_since_last_response < self.response_interval:
            return None

        observable_state = mind.get_observable_state()

        # Decide whether to respond based on the child's state and needs
        should_respond = self._should_respond(observable_state)

        if not should_respond:
            return None

        # Construct situation description from observable state
        situation = self._construct_situation(observable_state)

        # Determine appropriate response type based on needs and state
        response_type = self._determine_response_type(observable_state)

        # Get developmental stage for appropriate response style
        stage = observable_state.developmental_stage
        stage_name = stage.name

        # Select development focus area
        development_focus = self._select_development_focus(observable_state)

        # Choose technique based on focus area and stage
        technique = self._select_technique(stage, development_focus)

        # Generate response using templates or LLM
        response = self._generate_response(
            situation=situation,
            observable_state=observable_state,
            response_type=response_type,
            development_focus=development_focus,
            technique=technique,
        )

        if response:
            # Add to interaction history
            self.interaction_history.append({
                "observation": observable_state.to_dict(),
                "response": response.to_dict(),
                "timestamp": datetime.now().isoformat(),
            })

            # Limit history size
            if len(self.interaction_history) > 100:
                self.interaction_history = self.interaction_history[-100:]

            self.last_response_time = current_time
            return response

        return None

    def _should_respond(self, state: ObservableState) -> bool:
        """Determine whether the mother should respond to the current state.
        
        Args:
            state: Observable state of the mind
            
        Returns:
            True if the mother should respond, False otherwise

        """
        # Always respond to strong needs
        for need, intensity in state.expressed_needs.items():
            if intensity > 0.7:
                return True

        # Always respond to strong negative emotions
        for emotion in state.recent_emotions:
            if emotion.name.value in ["fear", "sadness", "anger"] and emotion.intensity > 0.6:
                return True

        # Always respond to vocalizations
        if state.vocalization:
            return True

        # Respond based on developmental stage
        if state.developmental_stage == DevelopmentalStage.INFANT:
            # Infants need frequent attention
            return random.random() < 0.8
        if state.developmental_stage == DevelopmentalStage.TODDLER:
            # Toddlers need regular attention but some independence
            return random.random() < 0.6
        if state.developmental_stage == DevelopmentalStage.CHILD:
            # Children need moderate attention and more independence
            return random.random() < 0.4
        # Adolescents and mature minds need space but still connection
        return random.random() < 0.3

    def _construct_situation(self, state: ObservableState) -> str:
        """Convert observable state into a natural description for the LLM.
        
        Args:
            state: Observable state of the mind
            
        Returns:
            Natural language description of the situation

        """
        description = []

        # Describe developmental stage
        description.append(state.get_developmental_description())

        # Describe apparent mood
        mood_desc = "seems content"
        if state.apparent_mood < -0.5:
            mood_desc = "appears very distressed"
        elif state.apparent_mood < -0.2:
            mood_desc = "appears somewhat upset"
        elif state.apparent_mood > 0.5:
            mood_desc = "looks very happy"
        elif state.apparent_mood > 0.2:
            mood_desc = "looks cheerful"
        description.append(f"The child {mood_desc}.")

        # Describe energy level
        if state.energy_level < 0.3:
            description.append("They seem tired and low in energy.")
        elif state.energy_level > 0.7:
            description.append("They are very energetic and active.")

        # Describe focus/attention
        if state.current_focus:
            description.append(f"They are focused on {state.current_focus}.")

        # Describe recent emotions
        if state.recent_emotions:
            emotion_desc = ", ".join(
                f"showing {e.name.value} (intensity: {e.intensity:.1f})"
                for e in state.recent_emotions
            )
            description.append(f"Recently, they have been {emotion_desc}.")

        # Describe expressed needs
        if state.expressed_needs:
            needs_desc = ", ".join(
                f"seeking {need} (intensity: {intensity:.1f})"
                for need, intensity in state.expressed_needs.items()
                if intensity > 0.4  # Only include significant needs
            )
            if needs_desc:
                description.append(f"The child appears to be {needs_desc}.")

        # Describe vocalization
        if state.vocalization:
            description.append(f"The child {state.vocalization}.")

        # Describe behaviors
        if state.age_appropriate_behaviors:
            behavior_desc = ", ".join(state.age_appropriate_behaviors)
            description.append(f"Current behaviors: {behavior_desc}.")

        return " ".join(description)

    def _determine_response_type(self, state: ObservableState) -> str:
        """Determine the appropriate response type based on observed state.
        
        Args:
            state: Observable state of the mind
            
        Returns:
            Response type (comfort, play, teach, etc.)

        """
        # Check for highest intensity need
        highest_need = None
        highest_intensity = 0

        for need, intensity in state.expressed_needs.items():
            if intensity > highest_intensity:
                highest_need = need
                highest_intensity = intensity

        # If there's a strong need, respond to it
        if highest_need and highest_intensity > 0.5:
            if highest_need in ["comfort", "play", "rest"]:
                return highest_need

        # Check emotional state
        negative_emotions = ["sadness", "fear", "anger", "disgust"]
        for emotion in state.recent_emotions:
            if emotion.name.value in negative_emotions and emotion.intensity > 0.5:
                return "comfort"

        # If child seems receptive to learning
        if state.energy_level > 0.3 and state.apparent_mood > -0.3:
            # Higher probability of teaching at more advanced stages
            teach_probability = 0.3 + (state.developmental_stage.value * 0.1)
            if random.random() < teach_probability:
                return "teach"

        # If child is high energy, suggest play
        if state.energy_level > 0.6:
            return "play"

        # If child has low energy, suggest rest
        if state.energy_level < 0.3:
            return "rest"

        # Default to comfort for infants, teach for older stages
        if state.developmental_stage == DevelopmentalStage.INFANT:
            return "comfort"
        return "teach"

    def _select_development_focus(self, state: ObservableState) -> str:
        """Select an area of development to focus on for this interaction.
        
        Args:
            state: Observable state of the mind
            
        Returns:
            Development focus area (language, emotional, cognitive, physical)

        """
        # All developmental stages need balanced focus
        focus_areas = ["language", "emotional", "cognitive", "physical"]

        # Adjust probabilities based on developmental stage
        weights = {
            "language": 1.0,
            "emotional": 1.0,
            "cognitive": 1.0,
            "physical": 1.0,
        }

        # For infants, emotional and physical development are most important
        if state.developmental_stage == DevelopmentalStage.INFANT:
            weights["emotional"] = 1.5
            weights["physical"] = 1.5
            weights["cognitive"] = 0.7

        # For toddlers, language development becomes more important
        elif state.developmental_stage == DevelopmentalStage.TODDLER:
            weights["language"] = 1.5
            weights["physical"] = 1.3

        # For children, cognitive development becomes more important
        elif state.developmental_stage == DevelopmentalStage.CHILD:
            weights["cognitive"] = 1.5
            weights["language"] = 1.3

        # For adolescents, emotional development becomes crucial again
        elif state.developmental_stage == DevelopmentalStage.ADOLESCENT:
            weights["emotional"] = 1.5
            weights["cognitive"] = 1.3

        # Calculate total weight
        total_weight = sum(weights.values())

        # Normalize weights
        for area in weights:
            weights[area] /= total_weight

        # Weighted random selection
        r = random.random()
        cumulative = 0
        for area, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return area

        # Fallback
        return random.choice(focus_areas)

    def _select_technique(self, stage: DevelopmentalStage, focus: str) -> str:
        """Select a specific developmental technique for the focus area and stage.
        
        Args:
            stage: Developmental stage
            focus: Focus area
            
        Returns:
            Specific technique to use

        """
        techniques = self.developmental_techniques.get(stage, {}).get(focus, [])

        if not techniques:
            # Fallback if no techniques for this stage/focus
            all_techniques = []
            for s in self.developmental_techniques:
                for f in self.developmental_techniques[s]:
                    all_techniques.extend(self.developmental_techniques[s][f])
            return random.choice(all_techniques) if all_techniques else "general interaction"

        return random.choice(techniques)

    def _generate_response(
        self,
        situation: str,
        observable_state: ObservableState,
        response_type: str,
        development_focus: str,
        technique: str,
    ) -> Optional[MotherResponse]:
        """Generate a response to the observed situation.
        
        Args:
            situation: Description of the observable situation
            observable_state: Observable state of the mind
            response_type: Type of response to generate
            development_focus: Developmental focus area
            technique: Specific technique to use
            
        Returns:
            Generated response or None if generation fails

        """
        # Try to use a template first (more efficient than LLM for simple responses)
        template_response = self._get_template_response(
            observable_state.developmental_stage,
            response_type,
        )

        if template_response and random.random() < 0.7:  # 70% chance to use template
            return MotherResponse(
                understanding=f"Child appears to need {response_type}.",
                response=template_response,
                action=response_type,
                development_focus=development_focus,
            )

        # Otherwise, use LLM for more nuanced responses
        try:
            # Construct the prompt
            stage_name = observable_state.developmental_stage.name.capitalize()

            system_prompt = f"""You are a nurturing and attentive mother figure. You can only perceive external
            behaviors and must respond based on what you observe, not internal states.
            Your responses should be caring, supportive, and appropriate for the child's developmental stage.
            
            The child is currently at the {stage_name} developmental stage.
            You are focusing on {development_focus} development using {technique} techniques.
            The appropriate response type is: {response_type}.
            
            Your response should include:
            1. A brief understanding of what the child needs based on observations
            2. A nurturing verbal response that's appropriate for their developmental stage
            3. A specific action you're taking (comfort, teach, play, etc.)
            
            Keep your response concise, warm, and developmentally appropriate.
            """

            user_prompt = f"Observation of child: {situation}"

            # Get response from LLM with structured output
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                structured_output=True,
            )

            if response and "understanding" in response and "response" in response and "action" in response:
                # Create a MotherResponse from the LLM response
                mother_response = MotherResponse(
                    understanding=response["understanding"],
                    response=response["response"],
                    action=response["action"],
                    development_focus=development_focus,
                )

                return mother_response

            # Fallback to template if LLM fails
            logger.warning("LLM response did not contain required fields, falling back to template")
            return self._fallback_response(
                observable_state.developmental_stage,
                response_type,
                development_focus,
            )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e!s}")
            return self._fallback_response(
                observable_state.developmental_stage,
                response_type,
                development_focus,
            )

    def _get_template_response(self, stage: DevelopmentalStage, response_type: str) -> Optional[str]:
        """Get a template response for the given stage and type.
        
        Args:
            stage: Developmental stage
            response_type: Type of response
            
        Returns:
            Template response or None if no suitable template

        """
        stage_name = stage.name

        if stage_name in self.response_templates and response_type in self.response_templates[stage_name]:
            templates = self.response_templates[stage_name][response_type]
            if templates:
                return random.choice(templates)

        return None

    def _fallback_response(
        self,
        stage: DevelopmentalStage,
        response_type: str,
        development_focus: str,
    ) -> MotherResponse:
        """Generate a fallback response when other methods fail.
        
        Args:
            stage: Developmental stage
            response_type: Type of response
            development_focus: Developmental focus area
            
        Returns:
            Fallback response

        """
        # Use template if available
        template = self._get_template_response(stage, response_type)

        if template:
            return MotherResponse(
                understanding=f"Child appears to need {response_type}.",
                response=template,
                action=response_type,
                development_focus=development_focus,
            )

        # Emergency fallbacks based on stage
        stage_name = stage.name
        if stage == DevelopmentalStage.INFANT:
            return MotherResponse(
                understanding="Baby needs attention.",
                response="There, there. Mommy's here for you.",
                action="comfort",
                development_focus=development_focus,
            )
        if stage == DevelopmentalStage.TODDLER:
            return MotherResponse(
                understanding="Toddler needs interaction.",
                response="Let's play with your favorite toy!",
                action="play",
                development_focus=development_focus,
            )
        if stage == DevelopmentalStage.CHILD:
            return MotherResponse(
                understanding="Child could benefit from learning.",
                response="Would you like to learn something new today?",
                action="teach",
                development_focus=development_focus,
            )
        return MotherResponse(
            understanding="Child needs supportive presence.",
            response="I'm here if you'd like to talk or need anything.",
            action="comfort",
            development_focus=development_focus,
        )
