"""
Language interface for natural language communication.

Generates explanations, responses to queries, and communicates
decisions in natural language.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LanguageInterface:
    """
    Natural language communication for the robot brain.
    
    Generates:
    - Explanations of decisions
    - Responses to user queries
    - Mission summaries
    - Status reports
    """
    
    def __init__(self):
        """Initialize language interface."""
        self.conversation_history: List[str] = []
    
    def explain_decision(
        self,
        decision_type: str,
        action: str,
        confidence: float,
        reason: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate an explanation of a decision.
        
        Args:
            decision_type: Type of decision
            action: Action to take
            confidence: Confidence level [0, 1]
            reason: Reason for decision
            context: Optional context
            
        Returns:
            Natural language explanation
        """
        explanation = f"I'm {self._confidence_to_language(confidence)} that I should {action}. "
        explanation += f"This is because {reason}."
        
        if context:
            explanation += f" Given the context: {context}"
        
        return explanation
    
    def summarize_mission(
        self,
        goal: str,
        success: bool,
        duration: float,
        observations: List[str],
        outcome: str = "",
    ) -> str:
        """Generate a summary of a completed mission."""
        status = "successfully completed" if success else "attempted"
        summary = f"Mission '{goal}' was {status} in {duration:.1f} seconds.\n"
        
        if observations:
            summary += "Key observations:\n"
            for obs in observations[:3]:  # First 3
                summary += f"- {obs}\n"
        
        if outcome:
            summary += f"Result: {outcome}"
        
        return summary
    
    def query_response(
        self,
        query: str,
        relevant_context: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            query: User's question
            relevant_context: Relevant information from memory
            
        Returns:
            Response text
        """
        response = f"Regarding your question '{query}': "
        
        if relevant_context:
            response += "Based on my knowledge, " + relevant_context[0]
        else:
            response += "I don't have relevant information to answer that directly."
        
        return response
    
    def generate_status_report(
        self,
        battery_level: float,
        errors: List[str],
        current_goal: Optional[str] = None,
        nearby_obstacles: Optional[List[str]] = None,
    ) -> str:
        """Generate a status report."""
        report = "Status Report:\n"
        report += f"- Battery: {battery_level*100:.1f}%\n"
        
        if current_goal:
            report += f"- Current Goal: {current_goal}\n"
        
        if errors:
            report += "- Issues:\n"
            for error in errors:
                report += f"  - {error}\n"
        
        if nearby_obstacles:
            report += "- Nearby Obstacles:\n"
            for obs in nearby_obstacles:
                report += f"  - {obs}\n"
        
        return report
    
    def apologize_and_recover(self, error_type: str) -> str:
        """Generate an apologetic message and recovery explanation."""
        messages = {
            "collision": "I apologize - I collided with an obstacle. I'm recovering by backing up and finding an alternate path.",
            "sensor_failure": "I apologize - a sensor has temporarily failed. I'm switching to degraded-mode operation.",
            "navigation_failure": "I apologize - I couldn't reach the destination. I'm reassessing the navigation plan.",
            "unknown_error": "I apologize - an unexpected error occurred. I'm stopping for safety and waiting for assistance.",
        }
        return messages.get(error_type, messages["unknown_error"])
    
    def _confidence_to_language(self, confidence: float) -> str:
        """Convert confidence value to language."""
        if confidence >= 0.9:
            return "very confident"
        elif confidence >= 0.7:
            return "fairly confident"
        elif confidence >= 0.5:
            return "somewhat confident"
        else:
            return "not very confident"
    
    def add_to_history(self, message: str, role: str = "robot") -> None:
        """Add message to conversation history."""
        self.conversation_history.append(f"{role}: {message}")
    
    def get_conversation_summary(self, num_messages: int = 5) -> str:
        """Get summary of recent conversation."""
        recent = self.conversation_history[-num_messages:]
        return "\n".join(recent)


# Prompt templates for LLM integration
LLM_PROMPTS = {
    "explain_decision": """You are a robot brain explaining a decision to a human.
Decision Type: {decision_type}
Proposed Action: {action}
Confidence: {confidence}
Reasoning: {reason}

Explain this decision in 1-2 sentences in a friendly, natural way.""",
    
    "plan_mission": """You are a robot planning a mission.
Goal: {goal}
Current Context: {context}
Constraints: {constraints}

Create a step-by-step plan to achieve this goal. Be specific and practical.""",
    
    "recovery_strategy": """You are planning recovery from a failure.
Failure: {failure}
Context: {context}

What steps should the robot take to recover? Be specific and prioritize safety.""",
    
    "status_report": """You are a robot summarizing its status.
Battery: {battery}%
Errors: {errors}
Current Goal: {goal}
Nearby: {obstacles}

Generate a brief, natural status update for the human operator.""",
}


def get_prompt_template(template_name: str) -> Optional[str]:
    """Get a prompt template for the LLM."""
    return LLM_PROMPTS.get(template_name)
