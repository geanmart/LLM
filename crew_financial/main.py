from crewai import Crew

from tasks import Tasks
from agents import Agents
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "NA"
#a

llm = ChatOpenAI(
model = "mixtral_q4",
base_url = "http://localhost:11434/v1")



tasks = Tasks()
agents = Agents()

# Create Agents
history_agent = agents.history_agent(llm=llm)
summa_agent = agents.summarize_agent(llm=llm)
review_agent = agents.review_agent(llm=llm)

# Define Tasks for each agent
hist_task = tasks.history_customer_task(history_agent)
summ_task = tasks.summarization_task(summa_agent)
review_task = tasks.review_task(review_agent)

# Instantiate the crew with a sequential process
crew = Crew(
    agents=[history_agent, summa_agent, review_agent],
    tasks=[
        hist_task,
        summ_task,
        review_task]
)

# Kick off the process
result = crew.kickoff()

print(result)