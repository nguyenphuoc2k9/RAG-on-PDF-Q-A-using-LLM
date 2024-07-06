### Install the required dependencies:

  

```bash
! pip install -q transformers==4.41.2
! pip install -q bitsandbytes==0.43.1
! pip install -q accelerate==0.31.0
! pip install -q langchain==0.2.5
! pip install -q langchainhub==0.1.20
! pip install -q langchain-chroma==0.1.1
! pip install -q langchain-community==0.2.5
! pip install -q langchain-openai==0.1.9
! pip install -q langchain_huggingface==0.0.3
! pip install -q chainlit==1.1.304
! pip install -q python-dotenv==1.0.1
! pip install -q pypdf==4.2.0
! npm install -g localtunnel
! pip install -q numpy==1.24.4
```

  

### Import:

  
  

  

```bash
import chainlit as cl
import torch
from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer , AutoModelForCausalLM , pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
```
### Define text splitter and embedding:
```bash
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

embedding = HuggingFaceEmbeddings()
```
### Vectorization:
1. File processing
```bash
def  process_file(file:AskFileResponse):
	if  file.type =="text/plain":
	Loader = TextLoader
	elif  file.type =="application/pdf":
	Loader = PyPDFLoader
	loader = Loader(file.path)
	documents = loader.load()
	docs = text_splitter.split_documents(documents)
	for i,doc in  enumerate(docs):
		doc.metadata["source"] = f"{i}.{doc.metadata['source']}"
	return docs
```
2. Create vector database
```bash
def  get_vector_db(file:AskFileResponse):
	docs = process_file(file)
	cl.user_session.set("docs",docs)
	db = Chroma.from_documents(docs,embedding)
	return db
```
```bash
YAML_PATH  =  "../safety_helmet_dataset/data.yaml"
EPOCHS  =  50
IMG_SIZE  =  640
BATCH_SIZE  =  12
model.train(data=YAML_PATH,
epochs  =  EPOCHS ,
batch  =  BATCH_SIZE ,
imgsz  =  IMG_SIZE )
```
### Define LLM model
```bash
def  get_huggingface_llm(model_name:str="lmsys/vicuna-7b-v1.5",max_new_token:int=512):
	nf4_config = BitsAndBytesConfig(
		load_in_4bits = True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=torch.bffloat16
	)
	model = AutoModelForCausalLM.from_pretrained(
	model_name,
	quantization_config=nf4_config,
	low_cpu_mem_usage=True
	)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model_pipeline = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		max_new_tokens=max_new_token,
		device_map="auto",
		pad_token_id=tokenizer.eos_token_id
	)
	llm = HuggingFacePipeline(pipeline=model_pipeline)
	return llm
```
### Initialize Chainlit UI:
1. On_chat_start function:
```bash
	welcome_message = """Welcome to the PDF QA! To get started :

2. Upload a PDF or text file

3. Ask a question about the file

"""
@cl.on_chat_start
async  def  on_chat_start():
files = None
while files is  None:
	files = await cl.AskFileMessage(content=welcome_message,accept=["text/plain","application/pdf"],max_size_mb=20,timeout=180).send()
	file = files[0]
	msg = cl.Message(content=f"Processing file {file.name}...",disable_feedback=True)
	await msg.send()
	vector_db = await cl.make_async(get_vector_db)(file)
	llm = await cl.make_async(get_huggingface_llm)()
	message_history = ChatMessageHistory()
	memory=ConversationBufferMemory(memory_key="chat_history",chat_memory=message_history,output_key="answer",return_messages=True)
	retriever = vector_db.as_retriever(search_type="mmr",search_kwargs={"k":3})
	chain = ConversationalRetrievalChain.from_llm(
		llm=llm,
		chain_type="stuff",
		retriever=retriever,
		memory=memory,
		return_source_documents=True,
	)
	msg.content = f"{file.name} processed. You can now ask questions!"
	await msg.update()
	cl.user_session.set("chain",chain)
```
2. On_message function

```bash
@cl.on_message
async  def  on_message(message:cl.Message):
	chain = cl.user_session.get("chain")
	cb = cl.AsyncLangChainCallbackHandler()
	res = await chain.ainvoke(message.content,callbacks=[cb])
	answer = res["answer"]
	source_documents = res["source_documents"]
	text_elements = []
	if source_documents:
		for source_idx,source_doc in  enumerate(source_documents):
			source_name = f"source_{source_idx}"
			text_elements.append(cl.Text(content=source_doc.page_content,name=source_name))
			sources_names = [text_el.name for text_el in text_elements]
			if source_names:
				answer += f"\nSources : {', '.join(sources_names)}"
			else:
				answer += "\nNo Sources found"
				msg = cl.Message(content=answer,elements=text_elements)
				await msg.send()
```
### Running the application
1. Run chainlit app
```bash
!chainlit run app.py --host 0.0.0.0 --port 8000 &>/content/logs.txt &
```
2. Public app
```bash
import urllib
print ("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8 ').strip("\n")  )
```
3. Access the website
```bash
!lt --port 8000 --subdomain aivn-simple-rag
```
