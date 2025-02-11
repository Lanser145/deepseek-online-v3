import streamlit as st
from huggingface_hub import InferenceClient
import json
import os
import time

# ======================
# CONFIGURACIÓN SEGURA
# ======================
st.set_page_config(
    page_title="Chatbot Pro",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Configuración desde variables de entorno
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "❌ HF_TOKEN no encontrado en variables de entorno!"

# ======================
# MODELO PRINCIPAL
# ======================
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# ======================
# CONFIGURACIÓN ACTUALIZADA
# ======================
MODEL_CONFIG = {
    "HuggingFaceH4/zephyr-7b-beta": {
        "max_tokens": 384,
        "temperature": 0.2,
        "top_p": 0.95,
    }
}

# ======================
# MANEJO DE DATOS
# ======================
CHATS_FILE = "chats_db.json"

def cargar_chats():
    try:
        if not os.path.exists(CHATS_FILE):
            return []

        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            contenido = f.read().strip()
            
            if not contenido:
                return []

            datos = json.loads(contenido)
            
            if not isinstance(datos, list):
                st.error("Estructura inválida del archivo. Reiniciando...")
                return []
                
            return datos

    except json.JSONDecodeError as e:
        st.error(f"Error decodificando JSON: {e}. Creando nuevo archivo...")
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []

    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.stop()

def guardar_chats(chats):
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                chats,
                f,
                indent=4,
                ensure_ascii=False,
                separators=(",", ": "),
                default=str
            )
        return True
    except Exception as e:
        st.error(f"Error guardando datos: {str(e)}")
        return False

# ======================
# NÚCLEO DE IA CORREGIDO
# ======================
class ChatEngine:
    def __init__(self):
        self.client = InferenceClient(token=HF_TOKEN)
    
    def generar_respuesta(self, prompt, historial):
        try:
            prompt = self._sanitizar_input(prompt)
            if not prompt:
                return "Por favor ingresa un mensaje válido"
            
            messages = self._construir_contexto(prompt, historial)
            
            response = self.client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                **MODEL_CONFIG[MODEL_NAME]
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return self._manejar_error(e)

    def _construir_contexto(self, prompt, historial):
        messages = [{
            "role": "system",
            "content": "Eres un asistente profesional. Responde en el mismo idioma del usuario."
        }]
        
        for msg in historial[-3:]:
            messages.append({
                "role": "user" if msg["rol"] == "user" else "assistant",
                "content": msg["contenido"][:500]
            })
        
        messages.append({"role": "user", "content": prompt[:1000]})
        return messages

    def _sanitizar_input(self, text):
        return text.strip().replace("\n", " ")[:1000]

    def _manejar_error(self, error):
        error_msg = str(error)
        if "429" in error_msg:
            time.sleep(15)
            return "⚠️ Inténtalo de nuevo en unos momentos..."
        elif "401" in error_msg:
            return "🔒 Error de autenticación - Verifica tu token"
        return f"🚨 Error técnico: {error_msg[:200]}"

# ======================
# INTERFAZ DE USUARIO
# ======================
def inicializar_sesion():
    if "chats" not in st.session_state:
        st.session_state.chats = cargar_chats()
    
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = ChatEngine()
    
    if not st.session_state.get("chat_actual"):
        st.session_state.chat_actual = st.session_state.chats[0] if st.session_state.chats else crear_nuevo_chat()

def crear_nuevo_chat():
    nuevo_chat = {
        "id": str(time.time_ns()),
        "titulo": f"Chat {len(st.session_state.chats) + 1}",
        "historial": []
    }
    st.session_state.chats.append(nuevo_chat)
    guardar_chats(st.session_state.chats)
    return nuevo_chat

def barra_lateral():
    with st.sidebar:
        st.header("Gestión de Chats")
        
        if st.button("➕ Nuevo Chat", use_container_width=True):
            st.session_state.chat_actual = crear_nuevo_chat()
            st.rerun()
        
        for chat in st.session_state.chats:
            btn_col, del_col = st.columns([8, 2])
            with btn_col:
                if st.button(
                    f"💬 {chat['titulo']}",
                    key=f"btn_{chat['id']}",
                    use_container_width=True
                ):
                    st.session_state.chat_actual = chat
            with del_col:
                if st.button("❌", key=f"del_{chat['id']}"):
                    eliminar_chat(chat)
        
        st.markdown("---")
        st.caption(f"v3.0 | Modelo: {MODEL_NAME.split('/')[-1]}")

def eliminar_chat(chat):
    try:
        st.session_state.chats.remove(chat)
        if st.session_state.chat_actual["id"] == chat["id"]:
            st.session_state.chat_actual = crear_nuevo_chat()
        guardar_chats(st.session_state.chats)
        st.rerun()
    except Exception as e:
        st.error(f"Error eliminando chat: {str(e)}")

def area_chat():
    st.title("🤖 Asistente Profesional")
    
    if not st.session_state.chat_actual["historial"]:
        st.info("¡Escribe tu primer mensaje para comenzar!")
    
    for msg in st.session_state.chat_actual["historial"]:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["contenido"])
    
    if prompt := st.chat_input("Escribe tu consulta..."):
        procesar_input(prompt)

def procesar_input(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            respuesta = st.session_state.chat_engine.generar_respuesta(
                prompt,
                st.session_state.chat_actual["historial"]
            )
            st.markdown(respuesta)
    
    actualizar_historial(prompt, respuesta)

def actualizar_historial(prompt, respuesta):
    st.session_state.chat_actual["historial"].extend([
        {"rol": "user", "contenido": prompt},
        {"rol": "assistant", "contenido": respuesta}
    ])
    guardar_chats(st.session_state.chats)

# ======================
# EJECUCIÓN
# ======================
inicializar_sesion()
barra_lateral()
area_chat()

# ======================
# ESTILOS
# ======================
st.markdown("""
<style>
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #1E293B;
        border: 1px solid #334155;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        width: calc(100% - 4rem);
    }
    
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)