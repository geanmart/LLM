from textwrap import dedent
from crewai import Task

class Tasks():
		def history_customer_task(self, agent):
				return Task(
						description=dedent(f"""\
								Analyze the provided customer past financial situation. Focus on understanding the customers debt over time.
								Compile a report summarizing these insights."""),
						expected_output=dedent("""\
								A comprehensive report detailing the customer financial health based on its historic debts."""),
						agent=agent
				)

		def summarization_task(self, agent):
				return Task(
						description=dedent(f"""\
								You will receive a call transcription between a bank employee and a customer. Summarize with attention to the detailed information about debt values and potential negociations between the employee and the customer."""),
						expected_output=dedent("""\
								Provide the summarization of the call transcription in Portuguese."""),
						agent=agent
				)

		def review_task(self, agent):
				return Task(
						description=dedent(f"""\
								Given the customers financial history and the conversation they had with a bank representative recently, assess and provide in detail the customers current status with the bank"""),
						expected_output=dedent("""\
								A detailed, summarization based on customer history and the recent call transcription."""),
						agent=agent
				)
