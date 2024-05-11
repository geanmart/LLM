from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "NA"
#a

llm = ChatOpenAI(
model = "mixtral_q4",
base_url = "http://localhost:11434/v1")

general_agent = Agent(role = "Summarization Analyst",
                  goal = """Provide the summarization of a call transcription.""",
                  backstory = """
                            Falante 0 (Atendente): Olá, bom dia! Aqui é o Carlos da Rede Brasil, prestando serviço para o Banco Itaú. Por gentileza, poderia confirmar seu nome?

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

                            Falante 1: Igualmente, até mais.""",
                  allow_delegation = False,
                  verbose = True,
                  llm = llm)
task = Task (description="""What was the call about ?""",
         agent = general_agent,
         expected_output="Summarization written in Portuguese with the main points of the call.") 

crew = Crew(
        agents=[general_agent],
        tasks=[task],
        verbose=2
    )
result = crew.kickoff()
print(result)