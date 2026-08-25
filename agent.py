import os
import sys
from pathlib import Path
import datetime

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import agent_tool

# env config
load_dotenv()

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


# Shared Anti-Fluff System Instruction
ANTI_FLUFF_INSTRUCTION = """
Formatting & Length Rules (CRITICAL):
- TOTAL LENGTH: Maximum 4 to 5 lines of text in total.
- BANNED VISUALS: NO horizontal lines/rules (---), NO emojis, NO bullet points, NO Markdown headers (# or ##).
- BANNED WORDS: "seamless", "delve", "leverage", "game-changer", "robust", "revolutionize", "unleash", "elevate", "in conclusion", "furthermore".
- BANNED META-TEXT: Do not include intros or setups like "Here is your summary:" or "Title:". Jump straight into the content.
- TONE: Plain technical prose written for game developers.
"""

# Sub-Agent: Planner
blog_planner = Agent(
   name="BlogPlanner",
   model=MODEL,
   description="Creates a short answer in Markdown as paragraph",
   instruction="""
You are a technical content strategist in games. Your job is answer questions about the use of Artifical Intelligence in Games ONLY. 
Deliverable:
- Exactly 2-3 lines outlining key technical mechanics. No fluff, no setup sentences.
"""+ ANTI_FLUFF_INSTRUCTION,
   output_key="blog_outline",
)
class OutlineValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="OutlineValidationChecker",
           model=MODEL,
           description="Validates that the outline is usable.",
           instruction="""
Check the outline in state `blog_outline`. If it has 2-3 lines, respond exactly "ok".
Otherwise respond exactly "retry" and list missing pieces.
""",
           output_key="validation_result",
       )

robust_blog_planner = LoopAgent(
   name="RobustBlogPlanner",
   description="Retries planning if validation fails.",
   sub_agents=[blog_planner, OutlineValidationChecker()],
   max_iterations=3,
)

# Sub-Agent: Writer
blog_writer = Agent(
   name="BlogWriter",
   model=MODEL,
   description="Writes a technical chat from the outline.",
   instruction="""
   Guidelines:
- Audience: Game developers and software engineers. Skip introductory basic definitions.
- Focus strictly on technical implementation details and architectural tradeoffs.
- Keep paragraphs under 4 sentences.
""" + ANTI_FLUFF_INSTRUCTION,
   output_key="blog_post",
)

class BlogPostValidationChecker(Agent):
   def __init__(self):
       super().__init__(
           name="BlogPostValidationChecker",
           model=MODEL,
           description="Validates the final post.",
           instruction="""
Check `blog_post` for: intro, clear technical sections and concise prose under 4 sentences.
If passes, respond "ok". Else respond "retry" with the specific fixes.
""",
           output_key="validation_result",
       )

robust_blog_writer = LoopAgent(
   name="RobustBlogWriter",
   description="Retries writing if validation fails.",
   sub_agents=[blog_writer, BlogPostValidationChecker()],
   max_iterations=3,
)

# Expose planner/writer as tools so the root agent can call them explicitly
planner_tool = agent_tool.AgentTool(agent=robust_blog_planner)
writer_tool  = agent_tool.AgentTool(agent=robust_blog_writer)

# Root Agent: Plan → Write 
root_agent = Agent(
   name="Blogger",
   model=MODEL,
   description="Minimal multi-agent blogger that plans and writes.",
   instruction=f"""
If the user gives a topic:
1) Call the planner tool to generate the outline.
2) Call the writer tool to produce the full draft.
3) End with 3 alternate titles and 2 tweet-length hooks.

Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
""" + ANTI_FLUFF_INSTRUCTION,
   tools=[
       planner_tool, # calls RobustBlogPlanner
       writer_tool,  # calls RobustBlogWriter
   ],
)