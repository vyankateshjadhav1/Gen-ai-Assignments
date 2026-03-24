"""
Prompt Engineering Approaches Implementation and Analysis
==========================================================
This module demonstrates and compares various prompt engineering techniques:
1. Interview Approach
2. Chain of Thought (CoT)
3. Tree of Thought (ToT)
4. Zero-shot Prompting
5. Few-shot Prompting
"""

import json
from typing import List, Dict, Any
from datetime import datetime


class PromptEngineeringDemo:
    """
    A comprehensive demonstration of various prompt engineering approaches.
    """
    
    def __init__(self):
        self.results = {}
        
    # ==================== INTERVIEW APPROACH ====================
    
    def interview_approach(self, problem: str) -> Dict[str, Any]:
        """
        Interview Approach: Breaks down the problem through iterative questioning.
        Simulates a conversation where the AI asks clarifying questions before solving.
        
        Use Case: Complex problems requiring clarification and context gathering.
        """
        print("\n" + "="*70)
        print("INTERVIEW APPROACH")
        print("="*70)
        
        prompts = []
        
        # Step 1: Initial problem statement
        initial_prompt = f"""
Problem to solve: {problem}

As an expert interviewer, please ask 3-5 clarifying questions to better 
understand this problem before providing a solution.
"""
        prompts.append(("Initial", initial_prompt))
        
        # Step 2: Simulated clarifying questions
        clarifying_questions = f"""
Based on the problem: {problem}

Clarifying Questions:
1. What are the constraints or limitations?
2. What is the expected output format?
3. Are there any edge cases to consider?
4. What is the priority: speed, accuracy, or resource efficiency?
5. What is the scale of data we're working with?

Now, let's assume the following answers:
1. Standard computational constraints
2. Clear, structured output
3. Handle null/empty inputs
4. Balance between accuracy and speed
5. Small to medium datasets

With these clarifications, please provide a detailed solution.
"""
        prompts.append(("Clarification", clarifying_questions))
        
        # Step 3: Final solution with context
        solution_prompt = f"""
Given the problem: {problem}
With the clarifications provided above,

Please provide:
1. A step-by-step solution
2. Code implementation (if applicable)
3. Potential optimizations
4. Testing strategy
"""
        prompts.append(("Solution", solution_prompt))
        
        result = {
            "approach": "Interview Approach",
            "description": "Iterative questioning to gather context before solving",
            "prompts": prompts,
            "advantages": [
                "Clarifies ambiguous requirements",
                "Ensures comprehensive understanding",
                "Reduces misinterpretation",
                "Builds context incrementally"
            ],
            "disadvantages": [
                "Requires multiple interactions",
                "More time-consuming",
                "May need human intervention for responses"
            ],
            "best_for": [
                "Complex, ambiguous problems",
                "Requirements gathering",
                "Stakeholder engagement scenarios"
            ]
        }
        
        self._print_approach_details(result)
        return result
    
    # ==================== CHAIN OF THOUGHT (CoT) ====================
    
    def chain_of_thought(self, problem: str) -> Dict[str, Any]:
        """
        Chain of Thought (CoT): Encourages step-by-step reasoning.
        The model explicitly shows its reasoning process.
        
        Use Case: Mathematical problems, logical reasoning, multi-step tasks.
        """
        print("\n" + "="*70)
        print("CHAIN OF THOUGHT (CoT)")
        print("="*70)
        
        prompt = f"""
Problem: {problem}

Let's solve this step by step:

Step 1: Understand the problem
- Break down what we're asked to do
- Identify key components

Step 2: Plan the approach
- What method or algorithm should we use?
- What are the intermediate steps?

Step 3: Execute each step
- Show detailed working for each step
- Explain the reasoning at each stage

Step 4: Verify the solution
- Check if the answer makes sense
- Consider edge cases

Please work through this problem following these steps, showing all your 
reasoning and intermediate calculations.
"""
        
        result = {
            "approach": "Chain of Thought (CoT)",
            "description": "Step-by-step reasoning with explicit intermediate steps",
            "prompt": prompt,
            "advantages": [
                "Improves accuracy on complex problems",
                "Makes reasoning transparent and verifiable",
                "Helps catch errors in logic",
                "Better for multi-step problems"
            ],
            "disadvantages": [
                "Longer responses",
                "More tokens/cost",
                "May be overkill for simple problems"
            ],
            "best_for": [
                "Mathematical problems",
                "Logical reasoning tasks",
                "Multi-step procedures",
                "Problems requiring explanation"
            ],
            "example": self._cot_example()
        }
        
        self._print_approach_details(result)
        return result
    
    def _cot_example(self) -> Dict[str, str]:
        """Provides a concrete CoT example."""
        return {
            "problem": "If a store has 35 apples and sells 3/5 of them, then receives a new shipment of 28 apples, how many apples does it have?",
            "cot_solution": """
Let's solve this step by step:

Step 1: Calculate apples sold
- Total apples: 35
- Fraction sold: 3/5
- Apples sold = 35 × (3/5) = 35 × 3 / 5 = 105 / 5 = 21 apples

Step 2: Calculate remaining apples after sale
- Remaining = 35 - 21 = 14 apples

Step 3: Add new shipment
- New shipment: 28 apples
- Total = 14 + 28 = 42 apples

Final answer: The store has 42 apples.

Verification: Started with 35, sold 21 (leaves 14), added 28 (equals 42) ✓
"""
        }
    
    # ==================== TREE OF THOUGHT (ToT) ====================
    
    def tree_of_thought(self, problem: str) -> Dict[str, Any]:
        """
        Tree of Thought (ToT): Explores multiple reasoning paths simultaneously.
        Evaluates different approaches and selects the best one.
        
        Use Case: Complex problems with multiple solution paths, strategic planning.
        """
        print("\n" + "="*70)
        print("TREE OF THOUGHT (ToT)")
        print("="*70)
        
        prompt = f"""
Problem: {problem}

Let's explore multiple solution paths using Tree of Thought approach:

BRANCH 1: Analytical Approach
├─ Step 1.1: [First analytical step]
├─ Step 1.2: [Second analytical step]
└─ Evaluation: [Assess feasibility, pros/cons]

BRANCH 2: Creative Approach
├─ Step 2.1: [First creative alternative]
├─ Step 2.2: [Second creative step]
└─ Evaluation: [Assess feasibility, pros/cons]

BRANCH 3: Hybrid Approach
├─ Step 3.1: [Combine elements from above]
├─ Step 3.2: [Optimize the combination]
└─ Evaluation: [Assess feasibility, pros/cons]

EVALUATION CRITERIA:
- Efficiency
- Accuracy
- Scalability
- Implementation complexity

FINAL DECISION:
Based on the evaluations, select the best approach and explain why.
Provide the complete solution using the chosen path.
"""
        
        result = {
            "approach": "Tree of Thought (ToT)",
            "description": "Explores multiple reasoning paths, evaluates and selects best",
            "prompt": prompt,
            "advantages": [
                "Explores diverse solution strategies",
                "Can find optimal solutions",
                "Evaluates trade-offs explicitly",
                "Good for complex strategic problems"
            ],
            "disadvantages": [
                "Computationally expensive",
                "Requires multiple generations",
                "Can be overwhelming for simple problems",
                "Needs good evaluation criteria"
            ],
            "best_for": [
                "Strategic planning",
                "Complex optimization problems",
                "Creative problem solving",
                "When multiple approaches exist"
            ],
            "example": self._tot_example()
        }
        
        self._print_approach_details(result)
        return result
    
    def _tot_example(self) -> Dict[str, str]:
        """Provides a concrete ToT example."""
        return {
            "problem": "Design a system to reduce website load time",
            "tot_solution": """
BRANCH 1: Frontend Optimization
├─ Minify CSS/JS
├─ Image optimization
└─ Evaluation: Quick wins, 20-30% improvement, easy implementation

BRANCH 2: Backend Optimization
├─ Database indexing
├─ Query optimization
└─ Evaluation: 30-40% improvement, medium complexity

BRANCH 3: Infrastructure Upgrade
├─ CDN implementation
├─ Caching layers
└─ Evaluation: 50-60% improvement, high initial cost

SELECTED: Hybrid approach combining Branch 1 (immediate) + Branch 2 (medium-term)
Rationale: Best ROI with manageable complexity
"""
        }
    
    # ==================== ZERO-SHOT PROMPTING ====================
    
    def zero_shot_prompting(self, task: str) -> Dict[str, Any]:
        """
        Zero-shot Prompting: Direct task instruction without examples.
        The model relies solely on its training to understand and execute.
        
        Use Case: Simple, clear tasks; when examples aren't available.
        """
        print("\n" + "="*70)
        print("ZERO-SHOT PROMPTING")
        print("="*70)
        
        prompt = f"""
Task: {task}

Please complete this task directly.
"""
        
        examples = [
            {
                "task": "Classify the sentiment: 'I absolutely loved this movie!'",
                "prompt": "Classify the sentiment of this text: 'I absolutely loved this movie!'",
                "expected_output": "Positive"
            },
            {
                "task": "Translate to French: 'Hello, how are you?'",
                "prompt": "Translate to French: 'Hello, how are you?'",
                "expected_output": "Bonjour, comment allez-vous?"
            },
            {
                "task": "Summarize: [Long text about AI]",
                "prompt": "Summarize this text in one sentence: [AI is transforming industries...]",
                "expected_output": "AI is revolutionizing multiple industries through automation and intelligence."
            }
        ]
        
        result = {
            "approach": "Zero-shot Prompting",
            "description": "Direct instruction without examples",
            "prompt": prompt,
            "examples": examples,
            "advantages": [
                "Simple and concise",
                "No need for examples",
                "Fast to implement",
                "Lower token usage"
            ],
            "disadvantages": [
                "May misinterpret task",
                "Less control over output format",
                "Lower accuracy on complex tasks",
                "Depends heavily on model capability"
            ],
            "best_for": [
                "Simple, well-defined tasks",
                "Common tasks the model knows well",
                "When examples aren't available",
                "Quick prototyping"
            ]
        }
        
        self._print_approach_details(result)
        return result
    
    # ==================== FEW-SHOT PROMPTING ====================
    
    def few_shot_prompting(self, task: str) -> Dict[str, Any]:
        """
        Few-shot Prompting: Provides examples before the actual task.
        The model learns the pattern from examples.
        
        Use Case: Tasks requiring specific format, style, or pattern.
        """
        print("\n" + "="*70)
        print("FEW-SHOT PROMPTING")
        print("="*70)
        
        # Example for sentiment classification
        sentiment_prompt = """
Classify the sentiment of the following texts as Positive, Negative, or Neutral.

Example 1:
Text: "This product exceeded my expectations!"
Sentiment: Positive

Example 2:
Text: "Terrible service, would not recommend."
Sentiment: Negative

Example 3:
Text: "The item arrived on time."
Sentiment: Neutral

Now classify:
Text: "Amazing quality and fast shipping!"
Sentiment: """
        
        # Example for text formatting
        formatting_prompt = """
Convert the following information into a structured format.

Example 1:
Input: John Smith works at Google as a Software Engineer since 2020
Output: {"name": "John Smith", "company": "Google", "role": "Software Engineer", "year": 2020}

Example 2:
Input: Sarah Johnson is a Data Scientist at Microsoft starting 2019
Output: {"name": "Sarah Johnson", "company": "Microsoft", "role": "Data Scientist", "year": 2019}

Example 3:
Input: Mike Brown, Product Manager, Amazon, 2021
Output: {"name": "Mike Brown", "company": "Amazon", "role": "Product Manager", "year": 2021}

Now convert:
Input: Emily Davis is a UX Designer at Apple beginning in 2022
Output: """
        
        result = {
            "approach": "Few-shot Prompting",
            "description": "Provides examples to demonstrate the desired pattern",
            "examples": {
                "sentiment_classification": sentiment_prompt,
                "structured_formatting": formatting_prompt
            },
            "advantages": [
                "Better accuracy than zero-shot",
                "Controls output format",
                "Teaches specific patterns",
                "More consistent results"
            ],
            "disadvantages": [
                "Requires good examples",
                "Uses more tokens",
                "Example selection is critical",
                "May not generalize well"
            ],
            "best_for": [
                "Tasks requiring specific format",
                "Style matching",
                "Pattern recognition tasks",
                "When you have good examples"
            ],
            "comparison_with_zero_shot": self._compare_zero_few_shot()
        }
        
        self._print_approach_details(result)
        return result
    
    def _compare_zero_few_shot(self) -> Dict[str, Any]:
        """Provides detailed comparison between zero-shot and few-shot."""
        return {
            "scenario": "Email categorization",
            "zero_shot": {
                "prompt": "Categorize this email: 'Meeting at 3pm tomorrow'",
                "characteristics": "Simple, direct, may vary in response",
                "accuracy": "60-70% (depending on task complexity)"
            },
            "few_shot": {
                "prompt": """
Example 1: "Project deadline extended" -> Category: Work
Example 2: "Dinner reservation confirmed" -> Category: Personal
Example 3: "Team meeting at 2pm" -> Category: Work

Categorize: "Meeting at 3pm tomorrow" """,
                "characteristics": "Learns from patterns, more consistent",
                "accuracy": "80-90% (improved through examples)"
            },
            "improvement": "15-25% accuracy increase with few-shot"
        }
    
    # ==================== COMPARISON AND ANALYSIS ====================
    
    def comprehensive_comparison(self) -> Dict[str, Any]:
        """
        Provides a comprehensive comparison of all approaches.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON")
        print("="*70)
        
        comparison = {
            "comparison_table": {
                "Interview Approach": {
                    "complexity": "Medium",
                    "interactions": "Multiple",
                    "token_usage": "High",
                    "accuracy": "High (with good questions)",
                    "use_time": "When requirements unclear"
                },
                "Chain of Thought": {
                    "complexity": "Medium",
                    "interactions": "Single",
                    "token_usage": "Medium-High",
                    "accuracy": "High (for reasoning tasks)",
                    "use_time": "For step-by-step problems"
                },
                "Tree of Thought": {
                    "complexity": "High",
                    "interactions": "Multiple branches",
                    "token_usage": "Very High",
                    "accuracy": "Very High (explores options)",
                    "use_time": "Complex strategic problems"
                },
                "Zero-shot": {
                    "complexity": "Low",
                    "interactions": "Single",
                    "token_usage": "Low",
                    "accuracy": "Medium (model-dependent)",
                    "use_time": "Simple, clear tasks"
                },
                "Few-shot": {
                    "complexity": "Low-Medium",
                    "interactions": "Single",
                    "token_usage": "Medium",
                    "accuracy": "High (with good examples)",
                    "use_time": "Pattern-matching tasks"
                }
            },
            "selection_guide": {
                "Simple classification": "Zero-shot or Few-shot",
                "Math problems": "Chain of Thought",
                "Strategic planning": "Tree of Thought",
                "Unclear requirements": "Interview Approach",
                "Format-specific output": "Few-shot",
                "Logical reasoning": "Chain of Thought",
                "Creative problem solving": "Tree of Thought",
                "Quick prototyping": "Zero-shot"
            },
            "performance_metrics": {
                "Speed": {
                    "1st": "Zero-shot",
                    "2nd": "Few-shot",
                    "3rd": "Chain of Thought",
                    "4th": "Interview Approach",
                    "5th": "Tree of Thought"
                },
                "Accuracy": {
                    "1st": "Tree of Thought",
                    "2nd": "Chain of Thought",
                    "3rd": "Few-shot",
                    "4th": "Interview Approach",
                    "5th": "Zero-shot"
                },
                "Cost_Efficiency": {
                    "1st": "Zero-shot",
                    "2nd": "Few-shot",
                    "3rd": "Chain of Thought",
                    "4th": "Interview Approach",
                    "5th": "Tree of Thought"
                }
            }
        }
        
        self._print_comparison(comparison)
        return comparison
    
    # ==================== PRACTICAL DEMONSTRATIONS ====================
    
    def demonstrate_all_approaches(self, problem: str):
        """
        Demonstrates all approaches on the same problem for comparison.
        """
        print("\n" + "="*70)
        print(f"DEMONSTRATING ALL APPROACHES ON: {problem}")
        print("="*70)
        
        results = {
            "problem": problem,
            "timestamp": datetime.now().isoformat(),
            "approaches": {}
        }
        
        # Run all approaches
        results["approaches"]["interview"] = self.interview_approach(problem)
        results["approaches"]["cot"] = self.chain_of_thought(problem)
        results["approaches"]["tot"] = self.tree_of_thought(problem)
        results["approaches"]["zero_shot"] = self.zero_shot_prompting(problem)
        results["approaches"]["few_shot"] = self.few_shot_prompting(problem)
        results["comparison"] = self.comprehensive_comparison()
        
        return results
    
    def practical_applications_analysis(self) -> Dict[str, Any]:
        """
        Analyzes practical applications of each approach in real-world scenarios.
        """
        print("\n" + "="*70)
        print("PRACTICAL APPLICATIONS ANALYSIS")
        print("="*70)
        
        applications = {
            "Interview Approach": {
                "real_world_scenarios": [
                    "Medical diagnosis systems (clarifying symptoms)",
                    "Legal document analysis (understanding context)",
                    "Customer service chatbots (gathering information)",
                    "Requirements engineering in software development"
                ],
                "industry_examples": {
                    "Healthcare": "Patient intake systems that ask follow-up questions",
                    "Finance": "Loan application assistants that clarify financial situations",
                    "Education": "Adaptive learning systems that assess student understanding"
                }
            },
            "Chain of Thought": {
                "real_world_scenarios": [
                    "Mathematical problem solvers",
                    "Code debugging assistants",
                    "Scientific reasoning tools",
                    "Legal argument construction"
                ],
                "industry_examples": {
                    "Education": "Tutoring systems that show work step-by-step",
                    "Finance": "Risk assessment with transparent reasoning",
                    "Engineering": "Design validation with clear justification"
                }
            },
            "Tree of Thought": {
                "real_world_scenarios": [
                    "Strategic business planning",
                    "Game AI (chess, go)",
                    "Drug discovery (exploring molecular combinations)",
                    "Architecture design (evaluating multiple designs)"
                ],
                "industry_examples": {
                    "Business": "Market entry strategy evaluation",
                    "Research": "Hypothesis exploration in scientific research",
                    "Creative": "Story plot development with multiple branches"
                }
            },
            "Zero-shot": {
                "real_world_scenarios": [
                    "Sentiment analysis",
                    "Simple translations",
                    "Text summarization",
                    "Spam detection"
                ],
                "industry_examples": {
                    "Social Media": "Content moderation",
                    "E-commerce": "Product review analysis",
                    "News": "Article categorization"
                }
            },
            "Few-shot": {
                "real_world_scenarios": [
                    "Custom format conversion",
                    "Style-specific content generation",
                    "Brand voice matching",
                    "Domain-specific classification"
                ],
                "industry_examples": {
                    "Marketing": "Ad copy generation in brand voice",
                    "Legal": "Contract clause extraction",
                    "Healthcare": "Medical report formatting"
                }
            }
        }
        
        self._print_applications(applications)
        return applications
    
    # ==================== HELPER METHODS ====================
    
    def _print_approach_details(self, result: Dict[str, Any]):
        """Pretty prints approach details."""
        print(f"\nApproach: {result['approach']}")
        print(f"Description: {result['description']}")
        print("\nAdvantages:")
        for adv in result.get('advantages', []):
            print(f"  ✓ {adv}")
        print("\nDisadvantages:")
        for dis in result.get('disadvantages', []):
            print(f"  ✗ {dis}")
        print("\nBest For:")
        for use in result.get('best_for', []):
            print(f"  • {use}")
    
    def _print_comparison(self, comparison: Dict[str, Any]):
        """Pretty prints comparison table."""
        print("\nCOMPARISON TABLE:")
        print("-" * 70)
        print(f"{'Approach':<20} {'Complexity':<12} {'Tokens':<10} {'Accuracy':<15}")
        print("-" * 70)
        
        for approach, metrics in comparison["comparison_table"].items():
            print(f"{approach:<20} {metrics['complexity']:<12} "
                  f"{metrics['token_usage']:<10} {metrics['accuracy']:<15}")
        
        print("\n\nSELECTION GUIDE:")
        print("-" * 70)
        for scenario, recommendation in comparison["selection_guide"].items():
            print(f"  {scenario}: {recommendation}")
        
        print("\n\nPERFORMANCE RANKINGS:")
        print("-" * 70)
        for metric, rankings in comparison["performance_metrics"].items():
            print(f"\n{metric}:")
            for rank, approach in rankings.items():
                print(f"  {rank}: {approach}")
    
    def _print_applications(self, applications: Dict[str, Any]):
        """Pretty prints practical applications."""
        for approach, details in applications.items():
            print(f"\n{approach}:")
            print("  Real-world Scenarios:")
            for scenario in details["real_world_scenarios"]:
                print(f"    • {scenario}")
            print("  Industry Examples:")
            for industry, example in details["industry_examples"].items():
                print(f"    {industry}: {example}")


# ==================== DEMONSTRATION EXAMPLES ====================

def run_sentiment_analysis_comparison():
    """
    Compares Zero-shot vs Few-shot for sentiment analysis.
    """
    print("\n" + "="*70)
    print("ZERO-SHOT VS FEW-SHOT: SENTIMENT ANALYSIS")
    print("="*70)
    
    test_cases = [
        "This product is amazing!",
        "Worst purchase ever.",
        "It's okay, nothing special.",
        "Exceeded all my expectations!",
        "Complete waste of money."
    ]
    
    print("\nZERO-SHOT APPROACH:")
    print("-" * 70)
    zero_shot_prompt = """
Classify the sentiment of the following text as Positive, Negative, or Neutral.

Text: "{text}"
Sentiment: """
    
    for text in test_cases:
        print(f"Text: {text}")
        print(f"Prompt: {zero_shot_prompt.format(text=text)}")
        print()
    
    print("\nFEW-SHOT APPROACH:")
    print("-" * 70)
    few_shot_prompt = """
Classify sentiment as Positive, Negative, or Neutral.

Example 1:
Text: "I love this product!"
Sentiment: Positive

Example 2:
Text: "Terrible quality."
Sentiment: Negative

Example 3:
Text: "It works as expected."
Sentiment: Neutral

Now classify:
Text: "{text}"
Sentiment: """
    
    for text in test_cases:
        print(f"Text: {text}")
        print(f"Using few-shot template with examples")
        print()
    
    print("\nEXPECTED IMPROVEMENTS:")
    print("-" * 70)
    print("✓ Few-shot provides consistent format")
    print("✓ Better handling of edge cases (e.g., 'okay' -> Neutral)")
    print("✓ Higher confidence in classifications")
    print("✓ Approximately 15-25% accuracy improvement")


def run_math_problem_comparison():
    """
    Demonstrates CoT vs standard approach for math problems.
    """
    print("\n" + "="*70)
    print("CHAIN OF THOUGHT: MATH PROBLEM SOLVING")
    print("="*70)
    
    problem = "A baker made 48 cupcakes. He sold 3/4 of them in the morning and 1/3 of the remaining in the afternoon. How many cupcakes are left?"
    
    print(f"\nProblem: {problem}")
    
    print("\n\nWITHOUT CoT (Direct):")
    print("-" * 70)
    direct_prompt = f"Solve: {problem}"
    print(direct_prompt)
    print("Expected: Direct answer without reasoning (may be incorrect)")
    
    print("\n\nWITH CoT (Step-by-step):")
    print("-" * 70)
    cot_prompt = f"""
Problem: {problem}

Let's solve this step by step:

Step 1: Calculate cupcakes sold in the morning
- Total cupcakes: 48
- Fraction sold: 3/4
- Sold in morning = 48 × 3/4 = 36 cupcakes

Step 2: Calculate remaining after morning
- Remaining = 48 - 36 = 12 cupcakes

Step 3: Calculate cupcakes sold in afternoon
- Fraction of remaining: 1/3
- Sold in afternoon = 12 × 1/3 = 4 cupcakes

Step 4: Calculate final remaining
- Final = 12 - 4 = 8 cupcakes

Answer: 8 cupcakes are left.

Verification: 48 - 36 - 4 = 8 ✓
"""
    print(cot_prompt)
    
    print("\nBENEFITS OF CoT:")
    print("-" * 70)
    print("✓ Transparent reasoning process")
    print("✓ Easier to identify errors")
    print("✓ Higher accuracy on multi-step problems")
    print("✓ Educational value (shows working)")


def main():
    """
    Main function to run all demonstrations.
    """
    demo = PromptEngineeringDemo()
    
    print("="*70)
    print("PROMPT ENGINEERING APPROACHES: COMPREHENSIVE ANALYSIS")
    print("="*70)
    print("\nThis demo covers:")
    print("1. Interview Approach")
    print("2. Chain of Thought (CoT)")
    print("3. Tree of Thought (ToT)")
    print("4. Zero-shot Prompting")
    print("5. Few-shot Prompting")
    print("\nWith comparisons, contrasts, and practical applications.")
    print("="*70)
    
    # Example problem for demonstration
    example_problem = "Design an efficient algorithm to find duplicate elements in an array"
    
    # Run all demonstrations
    results = demo.demonstrate_all_approaches(example_problem)
    
    # Show practical applications
    print("\n")
    applications = demo.practical_applications_analysis()
    
    # Specific comparisons
    print("\n")
    run_sentiment_analysis_comparison()
    
    print("\n")
    run_math_problem_comparison()
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    print("""
KEY TAKEAWAYS:

1. INTERVIEW APPROACH: Best for ambiguous problems
   - Use when requirements are unclear
   - Iterative refinement through questions
   - Higher interaction cost but better understanding

2. CHAIN OF THOUGHT (CoT): Best for reasoning tasks
   - Ideal for math, logic, multi-step problems
   - Improves accuracy significantly (20-30%)
   - Makes AI reasoning transparent

3. TREE OF THOUGHT (ToT): Best for complex decisions
   - Explores multiple solution paths
   - Optimal for strategic planning
   - Higher computational cost but better solutions

4. ZERO-SHOT: Best for simple, clear tasks
   - Fast and economical
   - Good for well-defined problems
   - Lower accuracy on complex tasks

5. FEW-SHOT: Best for pattern learning
   - Teaches specific formats through examples
   - 15-25% accuracy improvement over zero-shot
   - Ideal for consistent output formatting

SELECTION FRAMEWORK:
- Simple task + Clear instructions → Zero-shot
- Simple task + Specific format → Few-shot
- Complex reasoning + Need transparency → Chain of Thought
- Strategic decisions + Multiple options → Tree of Thought
- Unclear requirements + Need clarification → Interview

COST vs ACCURACY TRADE-OFF:
Zero-shot < Few-shot < CoT < Interview < ToT (cost)
Zero-shot < Interview < Few-shot < CoT < ToT (accuracy)
""")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
