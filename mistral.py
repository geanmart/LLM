import torch
from transformers import BitsAndBytesConfig

quantization_config  = BitsAndBytesConfig(
    load__in__4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
    )

model_id = 'mistralai/Mistral-7B-Instruct-v0.1'


## carregando o modelo instruct todo, com configurações de quantização

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_4bit = AutoModelForCausalLM(model_id,device_map="auto",quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline('text-generation', 
                    model=model_4bit, 
                    tokenizer=tokenizer,
                    use_cache=True,
                    device_map='auto',
                    max_length=1000,
                    do_sample=True,
                     top_k=5,
                     num_return_sequences=1,
                     eos_token_id=tokenizer.eos_token_id,
                     pad_token_id=tokenizer.pad_token_id)

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

llm = HuggingFacePipeline(pipeline=pipeline)


%%time 
template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few works from the context in portuguese.
Answer the question below from the context:
{context}
{question} [/INST]</s>
"""

question_p = """What is the name of the customer?"""
context_p = """Falante 0 (Atendente): Olá, bom dia! Aqui é o Carlos da Rede Brasil, prestando serviço para o Banco Itaú. Por gentileza, poderia confirmar seu nome?

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

Falante 1: Igualmente, até mais.
"""

prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt,llm=llm)
response = llm_chain.run({"question":question_p,"context":context_p})


del pipeline
del llm
del model_4bit
del tokenizer
del quantization_config

###GGUF ,para uso local em CPU

from langchain.llms import CTransformers
config = {'max_new_tokens': 1000, 'temperature': 0.7}

llm = CTransformers(model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    model_file="/home/gean/Documentos/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                    config=config)


%%time 
template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few works from the context in portuguese.
Answer the question below from the context:
{context}
{question} [/INST]</s>
"""

question_p = """What is the name of the customer?"""
context_p = """Falante 0 (Atendente): Olá, bom dia! Aqui é o Carlos da Rede Brasil, prestando serviço para o Banco Itaú. Por gentileza, poderia confirmar seu nome?

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

Falante 1: Igualmente, até mais.
"""
prompt = PromptTemplate(template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt,llm=llm)
response = llm_chain.run({"question":question_p,"context":context_p})

