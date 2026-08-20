from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Leemos las claves desde las variables de entorno (vacías por defecto para que Render las inyecte)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")

# Cerebro y Retriever conectado a la nube (Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="ujierpro", embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Almacén de historial de sesión
chat_histories = {}

# =====================================================================
# --- RUTAS PARA SERVIR TU PÁGINA WEB ---
# =====================================================================
@app.get("/")
def leer_index():
    return FileResponse("index.html")

@app.get("/manifest.json")
def leer_manifest():
    return FileResponse("manifest.json")

# =====================================================================
# --- RUTAS DE IA ---
# =====================================================================
template = """Eres un tutor experto en oposiciones de Ujieres. Responde a la pregunta del alumno basándote estrictamente en el contexto del temario proporcionado. Si hay historial previo de la conversación, tenlo en cuenta para mantener la coherencia.

Contexto del temario:
{context}

Historial reciente de la conversación:
{chat_history}

Pregunta del alumno: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/pregunta")
def recibir_pregunta(data: dict):
    pregunta = data.get("pregunta", "")
    session_id = "user_1"
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
        
    try:
        docs = retriever.invoke(pregunta)
        sources = list(set([doc.metadata.get("source", "Documento general") for doc in docs]))
        context_str = format_docs(docs)
        
        history_str = ""
        for h_q, h_a in chat_histories[session_id][-4:]:
            history_str += f"Humano: {h_q}\nTutor: {h_a}\n"
        if not history_str:
            history_str = "No hay historial previo."
            
        respuesta = chain.invoke({
            "context": context_str,
            "chat_history": history_str,
            "question": pregunta
        })
        
        chat_histories[session_id].append((pregunta, respuesta))
        
        return {"respuesta": respuesta, "fuentes": sources}
    except Exception as e:
        return {"respuesta": f"Error interno: {str(e)}", "fuentes": []}

@app.get("/generar-test")
def generar_test():
    try:
        temas_busqueda = ["Constitución", "ley", "derechos", "organización", "procedimiento", "administración"]
        docs = retriever.invoke(random.choice(temas_busqueda))
        contexto = format_docs(docs) if docs else "Constitución Española"
        
        test_prompt = ChatPromptTemplate.from_template(
            "Crea una pregunta tipo test sobre: {contexto}. "
            "Devuelve la respuesta estrictamente en JSON (sin markdown). Formato: "
            "{{\"pregunta\": \"...\", \"opciones\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"], \"correcta\": \"A\"}}"
        )
        test_chain = test_prompt | llm | StrOutputParser()
        respuesta = test_chain.invoke({"contexto": contexto})
        
        clean_json = respuesta.replace("`" * 3 + "json", "").replace("`" * 3, "").strip()
        return {"test": clean_json}
    except Exception as e:
        return {"test": f"{{\"pregunta\": \"Error generando test: {str(e)}\", \"opciones\": [], \"correcta\": \"\"}}"} 

@app.get("/generar-resumen-inteligente")
def generar_resumen_inteligente():
    try:
        temas = ["Constitución", "Tribunal Constitucional", "Cortes Generales", "Procedimiento Administrativo", "Gobierno", "Empleados Públicos"]
        tema_elegido = random.choice(temas)

        tavily_tool = TavilySearchResults(max_results=3)
        query_web = f"conceptos más preguntados preguntas trampa oposiciones {tema_elegido}"
        
        web_results = tavily_tool.invoke({"query": query_web})
        
        web_context = ""
        if isinstance(web_results, list):
            web_context = "\n".join([res.get("content", "") for res in web_results if isinstance(res, dict)])
        else:
            web_context = str(web_results)

        docs = retriever.invoke(f"{tema_elegido} {web_context[:100]}")
        local_context = format_docs(docs) if docs else "Sin datos locales."

        resumen_prompt = ChatPromptTemplate.from_template(
            "Eres un preparador de élite para oposiciones. Genera una 'Píldora de Alto Rendimiento'.\n\n"
            "TENDENCIAS EN FOROS: {web_context}\n"
            "TEORÍA OFICIAL DEL TEMARIO: {local_context}\n\n"
            "Instrucciones:\n"
            "1. Elabora un resumen ultra enfocado sobre '{tema_elegido}' cruzando las tendencias con la teoría oficial.\n"
            "2. Usa listas (<ul><li>). Resalta en negrita (<strong>) números críticos, plazos y mayorías.\n"
            "3. Incluye una sección destacada llamada '⚠️ POSIBLE PREGUNTA TRAMPA'.\n"
            "4. Finaliza con una pregunta tipo test relacionada con el resumen, indicando la solución.\n"
            "Formato de salida: Devuelve SOLO HTML limpio. No uses etiquetas <html> ni bloques markdown."
        )
        resumen_chain = resumen_prompt | llm | StrOutputParser()
        respuesta = resumen_chain.invoke({
            "web_context": web_context, 
            "local_context": local_context, 
            "tema_elegido": tema_elegido
        })
        
        clean_html = respuesta.replace("`" * 3 + "html", "").replace("`" * 3, "").strip()
        return {"resumen": clean_html}
    
    except Exception as e:
        return {"resumen": f"<p style='color:red;'>Error de Python: {str(e)}</p>"}