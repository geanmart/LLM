from crewai import Agent
from crewai_tools.tools import FileReadTool

file_read_tool = FileReadTool(
	file_path='/home/gean/Documentos/git_stuff/LLM/situacao_financeira.md',
	description='A tool to read the customer financial status example file.'
)

class Agents():
	def history_agent(self,llm):
		return Agent(
			role='Financial analyst',
			goal='Analyze the customer financial background and provide insights.',
			tools=[file_read_tool],
			backstory='Expert in evaluating the financial history of Banco Itaú customer',
			verbose=True,llm=llm
		)

	def summarize_agent(self,llm):
			return Agent(
				role='Summarization Analyst',
				goal='Provide the summarization of a call transcription.',
				#tools=[web_search_tool, seper_dev_tool, file_read_tool],
				backstory='''Falante 0 (Atendente): Olá, bom dia! Aqui é o Carlos da Rede Brasil, prestando serviço para o Banco Itaú. Por gentileza, poderia confirmar seu nome?

                            Falante 1 (Cliente): Oi, bom dia. Sou o Roberto, sim.

                            Falante 0: Obrigado, Roberto. Verificando aqui... o senhor possui uma pendência conosco e queríamos oferecer algumas opções para resolver isso. Tem um momento para discutirmos?

                            Falante 1: Ah, claro, pode falar.

                            Falante 0: Excelente! Temos uma oportunidade para pagamento à vista com um desconto de 40%, o que reduziria bastante o valor total. Caso prefira, também podemos arranjar um plano de pagamento parcelado com um desconto menor. Como lhe parece?

                            Falante 1: Hum, e quanto seria o desconto para o parcelado?

                            Falante 0: Para o parcelamento, podemos fazer com 20% de desconto, e o valor poderia ser dividido em até 12 vezes sem juros. Isso ajudaria a ajustar melhor no seu orçamento.

                            Falante 1: E qual seria o valor das parcelas então?

                            Falante 0: Um momento, vou calcular aqui... [barulho de teclas] Certo, as parcelas ficariam em torno de duzentos reais cada uma.

                            Falante 1: Tá, parece interessante. Vou precisar pensar um pouco, posso retornar a ligação mais tarde?

                            Falante 0: Claro, sem problemas! Mas gostaria de lembrar que essas condições são por tempo limitado. Posso reservar a oferta até o final do dia de hoje. Como deseja proceder?

                            Falante 1: Okay, eu ligo até o final do dia então. Obrigado, Carlos.

                            Falante 0: Por nada, Roberto. Estamos à disposição para qualquer dúvida. Tenha um ótimo dia!

                            Falante 1: Igualmente, até mais.''',
				verbose=True,llm=llm
			)

	def review_agent(self,llm):
			return Agent(
				role='Review Specialist',
				goal='Given the customers financial history and the conversation they had with a bank representative recently, assess and provide in detail the customers current status with the bank',
				#tools=[web_search_tool, seper_dev_tool, file_read_tool],
				backstory='A Expert in evaluating customer historical and current financial situation, with an eye for detail, ensuring every piece of content is clear, engaging, and precise.',
				verbose=True,llm=llm
			)